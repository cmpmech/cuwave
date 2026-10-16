from functools import lru_cache

import numpy as np
import numpy.typing as npt

STENCILS = ("taylor", "optimized")
DISPERSION_TOL = 1e-3  # phase velocity error an optimized row keeps over its band


def weights(R: int) -> npt.NDArray[np.float64]:
    """Central second-derivative weights of order 2R, `w[-R..R]`, via a Vandermonde solve."""
    x = np.arange(-R, R + 1)
    A = x[None, :] ** np.arange(2 * R + 1)[:, None]
    rhs = np.zeros(2 * R + 1)
    rhs[2] = 2.0
    return np.linalg.solve(A, rhs)


def cell_coefficients(R: int, stencil: str = "taylor") -> npt.NDArray[np.float64]:
    """Cumulative tail sums of `weights(R)`: the coefficients of the cell flux at radius `R`."""
    if stencil != "taylor":
        return _band_fit(R, stencil, staggered=False)
    w = weights(R)
    return np.array([w[R + 1 + k :].sum() for k in range(R)])


def staggered_weights(R: int, stencil: str = "taylor") -> npt.NDArray[np.float64]:
    """First-derivative weights of order 2R on the half offsets (2k - 1) / 2, `w[1..R]`."""
    if stencil != "taylor":
        return _band_fit(R, stencil, staggered=True)
    # antisymmetric taps, so only the odd moments constrain the R coefficients
    x = np.arange(1, R + 1) - 0.5
    A = x[None, :] ** (2 * np.arange(R)[:, None] + 1)
    rhs = np.zeros(R)
    rhs[0] = 0.5
    return np.linalg.solve(A, rhs)


@lru_cache
def _band_fit(R: int, stencil: str, staggered: bool) -> npt.NDArray[np.float64]:
    """Least-squares weights over the widest band whose phase velocity error stays under `DISPERSION_TOL`."""
    if stencil not in STENCILS:
        raise ValueError(f"unknown stencil {stencil!r}, expected one of {STENCILS}")
    k = np.arange(1, R + 1)
    for band in np.linspace(np.pi, 0.0, 400, endpoint=False):
        theta = np.linspace(band / 400, band, 400)
        # each symbol over its exact one, linear in the weights
        G = 2.0 * np.sin(np.outer(theta, k - 0.5)) / theta[:, None]
        if not staggered:
            G *= 2.0 * np.sin(theta / 2.0)[:, None] / theta[:, None]
        consistency = (2 * k - 1.0)[None]
        K = np.block([[G.T @ G, consistency.T], [consistency, np.zeros((1, 1))]])
        c = np.linalg.solve(K, np.append(G.sum(axis=0), 1.0))[:R]
        ratio = G @ c
        error = ratio - 1.0 if staggered else np.sqrt(ratio) - 1.0
        if np.abs(error).max() <= DISPERSION_TOL:
            return c


def preamble(space_order: int, stencil: str = "taylor") -> str:
    """CUDA preamble: `STENCIL_RADIUS` plus the coefficient tables, one row per radius."""
    R = space_order // 2
    flux = np.zeros((R, R))
    stag = np.zeros((R, R))
    for r in range(1, R + 1):
        # the graded wall rows stay Taylor, only the full radius is fitted
        row = stencil if r == R else "taylor"
        flux[r - 1, :r] = cell_coefficients(r, row)
        stag[r - 1, :r] = staggered_weights(r, row)

    def rows(table):
        return ", ".join(
            "{" + ", ".join(repr(float(v)) for v in row) + "}" for row in table
        )

    return (
        f"#define STENCIL_RADIUS {R}\n"
        f"#define OP_COEFFS {{ {rows(flux)} }}\n"
        f"#define STAG_COEFFS {{ {rows(stag)} }}\n"
    )
