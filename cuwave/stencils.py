import numpy as np
import numpy.typing as npt


def weights(R: int) -> npt.NDArray[np.float64]:
    """Central second-derivative weights of order 2R, `w[-R..R]`, via a Vandermonde solve."""
    x = np.arange(-R, R + 1)
    A = x[None, :] ** np.arange(2 * R + 1)[:, None]
    rhs = np.zeros(2 * R + 1)
    rhs[2] = 2.0
    return np.linalg.solve(A, rhs)


def cell_coefficients(R: int) -> npt.NDArray[np.float64]:
    """Cumulative tail sums of `weights(R)`: the coefficients of the cell flux at radius `R`."""
    w = weights(R)
    return np.array([w[R + 1 + k :].sum() for k in range(R)])


def staggered_weights(R: int) -> npt.NDArray[np.float64]:
    """First-derivative weights of order 2R on the half offsets (2k - 1) / 2, `w[1..R]`."""
    # antisymmetric taps, so only the odd moments constrain the R coefficients
    x = np.arange(1, R + 1) - 0.5
    A = x[None, :] ** (2 * np.arange(R)[:, None] + 1)
    rhs = np.zeros(R)
    rhs[0] = 0.5
    return np.linalg.solve(A, rhs)


def preamble(space_order: int) -> str:
    """CUDA preamble: `STENCIL_RADIUS` plus the coefficient tables, one row per radius."""
    R = space_order // 2
    flux = np.zeros((R, R))
    stag = np.zeros((R, R))
    for r in range(1, R + 1):
        flux[r - 1, :r] = cell_coefficients(r)
        stag[r - 1, :r] = staggered_weights(r)

    def rows(table):
        return ", ".join(
            "{" + ", ".join(repr(float(v)) for v in row) + "}" for row in table
        )

    return (
        f"#define STENCIL_RADIUS {R}\n"
        f"#define OP_COEFFS {{ {rows(flux)} }}\n"
        f"#define STAG_COEFFS {{ {rows(stag)} }}\n"
    )
