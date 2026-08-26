import numpy as np


def weights(R):
    """Central second-derivative finite-difference weights of order 2R, `w[-R..R]`, via a Vandermonde solve."""
    # collocated central second-derivative weights of order 2R, w[-R..R]
    x = np.arange(-R, R + 1)
    A = x[None, :] ** np.arange(2 * R + 1)[:, None]
    rhs = np.zeros(2 * R + 1)
    rhs[2] = 2.0
    return np.linalg.solve(A, rhs)


def face_coefficients(R):
    """Cumulative tail sums of `weights(R)`, the reduced-order coefficients used near a boundary face."""
    w = weights(R)
    return np.array([w[R + 1 + k :].sum() for k in range(R)])


def preamble(space_order):
    """CUDA preamble defining `STENCIL_RADIUS` and the `OP_COEFFS` table of face coefficients for every radius up to `space_order // 2`."""
    R = space_order // 2
    table = np.zeros((R, R))
    for r in range(1, R + 1):
        table[r - 1, :r] = face_coefficients(r)
    rows = ", ".join(
        "{" + ", ".join(repr(float(v)) for v in row) + "}" for row in table
    )
    return f"#define STENCIL_RADIUS {R}\n#define OP_COEFFS {{ {rows} }}\n"
