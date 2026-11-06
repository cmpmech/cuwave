from functools import lru_cache
from typing import NamedTuple

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


class Closure(NamedTuple):
    """The low wall's corner of a staggered SBP pair on h = 1, the high one its negated half turn."""

    node: tuple[tuple[float, ...], ...]  # (b, b + 1) Q+, the wall point first
    node_norm: tuple[float, ...]  # (b,) H+ of nodes 0..b-1, the wall node first
    half_norm: tuple[float, ...]  # (b,) H- of the half points 1/2..b-1/2


class StaggeredPair(NamedTuple):
    """The staggered SBP derivatives of an axis of `n` intervals on h = 1, wall points dropped."""

    node: npt.NDArray[np.float64]  # (n + 1, n) half points -> nodes
    half: npt.NDArray[np.float64]  # (n, n + 1) nodes -> half points
    node_norm: npt.NDArray[np.float64]  # (n + 1,) H+ on the nodes
    half_norm: npt.NDArray[np.float64]  # (n,) H- on the half points


def staggered_pair(space_order: int, n: int, stencil: str = "taylor") -> StaggeredPair:
    """The pair of Prochnow et al. 2017 on `n` intervals, its interior rows the `stencil` ones.

    Both walls are nodes, each with a half-grid wall point of zero norm read by the wall
    node's row alone, which a clamped wall holds at zero and a traction wall's zeroed
    normal stress never reads, so the point is dropped. `Q-` is `B - Q+^T`, so the pair
    sums by parts for any `stencil`: its rows replace the interior rows of `Q+` alone,
    the closure rows assuming the Taylor ones, and `Q-` follows.
    """
    if space_order not in CLOSURES:
        raise ValueError(
            f"no staggered closure of order {space_order}: {tuple(CLOSURES)}"
        )
    closure, R = CLOSURES[space_order], space_order // 2
    b = len(closure.node_norm)
    if n < 2 * b:
        raise ValueError(f"{n} intervals leave no room for two closures of {b} rows")

    # Q+ over the half grid with both wall points, the stencil's rows inside
    c = staggered_weights(R, stencil)
    qp = np.zeros((n + 1, n + 2))
    for i in range(n + 1):
        for k in range(1, R + 1):
            if i + k <= n + 1:
                qp[i, i + k] = c[k - 1]
            if i + 1 - k >= 0:
                qp[i, i + 1 - k] = -c[k - 1]

    # the corners, what they leave of their rows the Taylor row's
    taylor = staggered_weights(R)
    for i in range(b):
        for k in range(1, R + 1):
            qp[i, i + k] = taylor[k - 1]
            qp[n - i, n + 1 - i - k] = -taylor[k - 1]
    corner = np.array(closure.node)
    qp[:b, : b + 1] = corner
    qp[n + 1 - b :, n + 1 - b :] = -corner[::-1, ::-1]
    boundary = np.zeros_like(qp)
    boundary[0, 0], boundary[n, n + 1] = -1.0, 1.0
    qm = boundary.T - qp.T

    # both derivatives off the wall points
    node_norm, half_norm = closure_norms(space_order, n)
    node = qp[:, 1:-1] / node_norm[:, None]
    half = qm[1:-1] / half_norm[:, None]
    return StaggeredPair(node, half, node_norm, half_norm)


def closure_norms(
    space_order: int, n: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The pair's norms on `n` intervals, (n + 1,) on the nodes and (n,) on the half points."""
    closure = CLOSURES[space_order]
    b = len(closure.node_norm)
    node_norm, half_norm = np.ones(n + 1), np.ones(n)
    node_norm[:b], node_norm[n + 1 - b :] = closure.node_norm, closure.node_norm[::-1]
    half_norm[:b], half_norm[n - b :] = closure.half_norm, closure.half_norm[::-1]
    return node_norm, half_norm


def closure_band(space_order: int, stencil: str = "taylor") -> int:
    """Points from a wall within which a row of either derivative differs from the interior one."""
    return max(_closure_tables(space_order, stencil)[1][:2])


def closure_reach(space_order: int, stencil: str = "taylor") -> int:
    """Points from a wall the closure rows read: the half points of the node rows and the nodes of the half rows."""
    return max(_closure_tables(space_order, stencil)[1][2:])


@lru_cache
def _closure_tables(space_order: int, stencil: str) -> tuple[dict, tuple[int, ...]]:
    """The pair's low-wall rows the kernels read, and their counts and spans.

    Returns:
        the tables by name, and (node rows, half rows, node span, half span): the rows
        of each derivative off the interior one, the points they reach.
    """
    R, b = space_order // 2, len(CLOSURES[space_order].node_norm)
    n = 4 * (b + 4 * R)
    pair = staggered_pair(space_order, n, stencil)
    c = staggered_weights(R, stencil)

    # the rows that differ from the interior row centred on them
    def rows(matrix, offset):
        count = 0
        for i in range(n // 2):
            row = np.zeros(matrix.shape[1])
            for k in range(1, R + 1):
                if 0 <= i + k - 1 + offset < len(row):
                    row[i + k - 1 + offset] = c[k - 1]
                if 0 <= i - k + offset < len(row):
                    row[i - k + offset] -= c[k - 1]
            if not np.allclose(matrix[i], row, rtol=0.0, atol=1e-15):
                count = i + 1
        return count

    # at least one row each, so no table is empty
    node_rows, half_rows = max(rows(pair.node, 0), 1), max(rows(pair.half, 1), 1)
    node_span = int(np.flatnonzero(np.abs(pair.node[:node_rows]).sum(axis=0)).max()) + 1
    half_span = int(np.flatnonzero(np.abs(pair.half[:half_rows]).sum(axis=0)).max()) + 1
    tables = {
        "NODE_ROWS": pair.node[:node_rows, :node_span],
        "HALF_ROWS": pair.half[:half_rows, :half_span],
    }
    return tables, (node_rows, half_rows, node_span, half_span)


def preamble(space_order: int, stencil: str = "taylor") -> str:
    """CUDA preamble: `STENCIL_RADIUS`, the coefficient tables one row per radius, and the staggered closure."""
    R = space_order // 2
    flux = np.zeros((R, R))
    stag = np.zeros((R, R))
    for r in range(1, R + 1):
        # the graded wall rows stay Taylor, only the full radius is fitted
        row = stencil if r == R else "taylor"
        flux[r - 1, :r] = cell_coefficients(r, row)
        stag[r - 1, :r] = staggered_weights(r, row)

    def rows(table):
        if table.ndim == 1:
            return ", ".join(repr(float(v)) for v in table)
        return ", ".join("{" + rows(row) + "}" for row in table)

    code = (
        f"#define STENCIL_RADIUS {R}\n"
        f"#define OP_COEFFS {{ {rows(flux)} }}\n"
        f"#define STAG_COEFFS {{ {rows(stag)} }}\n"
    )
    if space_order not in CLOSURES:
        return code
    tables, counts = _closure_tables(space_order, stencil)
    names = ("NODE_CLOSURE", "HALF_CLOSURE", "NODE_SPAN", "HALF_SPAN")
    code += "".join(f"#define {k} {v}\n" for k, v in zip(names, counts))
    return code + "".join(f"#define {k}_C {{ {rows(v)} }}\n" for k, v in tables.items())


# Prochnow, O'Reilly, Dunham & Petersson 2017, https://doi.org/10.1016/j.compfluid.2017.03.015
CLOSURES = {
    2: Closure(
        node=((-1.0, 1.0),),
        node_norm=(0.5,),
        half_norm=(1.0,),
    ),
    4: Closure(
        node=(
            (
                -1.0,
                1.0863474151236463,
                -0.048972800926292664,
                -0.04940682870347156,
                0.012032214506117888,
            ),
            (
                0.0,
                -1.0912181712967304,
                1.020182291667565,
                0.106553819443748,
                -0.035517939814582664,
            ),
            (
                0.0,
                0.007306134259626,
                -1.020182291667565,
                1.020182291667565,
                -0.007306134259626,
            ),
            (
                0.0,
                -0.002435378086542,
                0.048972800926292664,
                -1.118995949074508,
                1.1141251929014242,
            ),
        ),
        node_norm=(
            0.3883101851851179,
            1.126736111111313,
            0.998263888888687,
            0.9866898148148822,
        ),
        half_norm=(
            1.0839120370371043,
            0.873263888888687,
            1.0434027777779797,
            0.999421296296229,
        ),
    ),
    6: Closure(
        node=(
            (
                -1.0,
                1.1394544837977862,
                -0.11190318831688334,
                -0.03679307743564517,
                -0.008806388435058028,
                0.024120360922567204,
                -0.006072190532767,
            ),
            (
                0.0,
                -1.135642086672916,
                1.020723413106962,
                0.12901498096987027,
                0.04871954028394196,
                -0.0882462187642217,
                0.025430371076363405,
            ),
            (
                0.0,
                0.014728358206091937,
                -1.0628624850408364,
                1.1157308563134456,
                -0.14782738341548576,
                0.11609496613710461,
                -0.03586431220032,
            ),
            (
                0.0,
                -0.01506154954283925,
                0.11105278167363838,
                -1.1157307986574199,
                1.1157308520205425,
                -0.11105286171832232,
                0.015061576224400562,
            ),
            (
                0.0,
                -0.01631208707591403,
                0.08569548827315682,
                -0.12901506745390884,
                -1.0716987136279488,
                1.1709042547946922,
                -0.044261374910077415,
            ),
            (
                0.0,
                0.012832881287791,
                -0.042706009696037585,
                0.03679310626365804,
                0.06856959317400804,
                -1.1722371680384867,
                1.1571642636757338,
            ),
        ),
        node_norm=(
            0.35421198253308606,
            1.1990789762234586,
            1.010175380886416,
            0.8578801746691397,
            1.1030043571098747,
            0.9756491285780251,
        ),
        half_norm=(
            1.152453960753951,
            0.6425913073413562,
            1.3188104408728434,
            0.8718145591271567,
            1.008971192658644,
            1.005358539246049,
        ),
    ),
    8: Closure(
        node=(
            (
                -1.0,
                1.159277909708591,
                -0.10225822161033948,
                -0.12235496146373043,
                0.06474950113711726,
                0.018000625588968488,
                -0.02778298490069253,
                0.013963295188980845,
                -0.0035951636488952684,
            ),
            (
                0.0,
                -1.08649665225212,
                0.7358131866897168,
                0.6469041496283849,
                -0.3324648837009965,
                -0.0336173293751992,
                0.10332516547775163,
                -0.036946531461095986,
                0.003482894993558249,
            ),
            (
                0.0,
                -0.08235828204281466,
                -0.736977485988319,
                0.31951655484386693,
                0.7032314422448357,
                -0.1703098567519811,
                -0.03624584851723378,
                -0.012075871596527,
                0.015219347808173,
            ),
            (
                0.0,
                0.02341742227992852,
                -0.147322512660754,
                0.16790851042687072,
                -0.4371788026324864,
                0.6774966552970128,
                -0.39016117877015266,
                0.1220854834403491,
                -0.016245577380768088,
            ),
            (
                0.0,
                -0.059899186728965506,
                0.49388162050978357,
                -1.4720495494260573,
                0.2555966584017443,
                0.13077614604961327,
                0.7765785014136136,
                -0.107508816058426,
                -0.017375374161306,
            ),
            (
                0.0,
                0.057499088529918,
                -0.2740066308298255,
                0.4978806316852063,
                -0.3763835826294999,
                -0.34419035975624124,
                0.5102303779913766,
                -0.08348589807751411,
                0.013153917729437,
            ),
            (
                0.0,
                -0.006126089943821456,
                0.010215000125073587,
                -0.019155436519664306,
                0.1508827394702544,
                -0.34375282227419424,
                -0.9464701792134959,
                1.1742811192535594,
                -0.02874709875485436,
            ),
            (
                0.0,
                -0.005314209550716,
                0.020655043764664007,
                -0.018649899174876958,
                -0.02843307229096885,
                0.06489939657916406,
                0.019398914375975886,
                -1.14119261699885,
                1.1595162796051317,
            ),
        ),
        node_norm=(
            0.338981562317581,
            1.2170844209197904,
            1.1833614859178774,
            0.2549326866095341,
            1.9558890726497251,
            0.3525760140821226,
            1.2319317827839134,
            0.9652429747194561,
        ),
        half_norm=(
            1.2070182602821766,
            0.368740688276087,
            1.8546141620235908,
            0.39080475586456204,
            1.1422350920190358,
            1.0954974451192663,
            0.9261047151630665,
            1.014984881252215,
        ),
    ),
}
