import cupy as cp
import cupy.typing as cpt

NAN = float("nan")


# -------------------------------------- helpers --------------------------------------
def _flatten_pair(
    field: cpt.NDArray, truth: cpt.NDArray
) -> tuple[cpt.NDArray, cpt.NDArray]:
    """Ravel field and truth to 1D, checking shapes match."""
    field, truth = cp.asarray(field).ravel(), cp.asarray(truth).ravel()
    if field.shape != truth.shape:
        raise ValueError(f"shape mismatch: {field.shape} against {truth.shape}")
    return field, truth


def _resolve_threshold(truth: cpt.NDArray, threshold: float | None = None) -> float:
    """Return threshold, defaulting to the midpoint of truth's range.

    The truth is always split at that midpoint, which is the natural cut for a
    two-material indicator, so no separate truth threshold is ever passed in.
    """
    return 0.5 * float(truth.min() + truth.max()) if threshold is None else threshold


def _ranked(
    field: cpt.NDArray, truth: cpt.NDArray
) -> tuple[cpt.NDArray, cpt.NDArray, int]:
    """Field and labels ordered by descending damage score. Also returns the positive count."""
    field, truth = _flatten_pair(field, truth)
    label = truth < _resolve_threshold(truth)
    order = cp.argsort(field)  # descending damage score (ascending indicator)
    return field[order], label[order], int(label.sum())


# ------------------------------ binary indicator fields ------------------------------
def confusion(
    field: cpt.NDArray, truth: cpt.NDArray, threshold: float | None = None
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) counts, `field` split at `threshold` and `truth` at its midpoint."""
    field, truth = _flatten_pair(field, truth)
    label = truth < _resolve_threshold(truth)
    predicted = field < _resolve_threshold(truth, threshold)
    tp = int(cp.count_nonzero(predicted & label))
    fp = int(cp.count_nonzero(predicted)) - tp
    fn = int(cp.count_nonzero(label)) - tp
    return tp, fp, fn, predicted.size - tp - fp - fn


def precision(
    field: cpt.NDArray, truth: cpt.NDArray, threshold: float | None = None
) -> float:
    """Share of the detected damage that is true, `tp / (tp + fp)`"""
    tp, fp, _, _ = confusion(field, truth, threshold)
    return tp / (tp + fp) if tp + fp else NAN


def recall(
    field: cpt.NDArray, truth: cpt.NDArray, threshold: float | None = None
) -> float:
    """Share of the true damage that is detected, `tp / (tp + fn)` (true positive rate)"""
    tp, _, fn, _ = confusion(field, truth, threshold)
    return tp / (tp + fn) if tp + fn else NAN


def false_positive_rate(
    field: cpt.NDArray, truth: cpt.NDArray, threshold: float | None = None
) -> float:
    """Share of the intact material flagged as damage, `fp / (fp + tn)`"""
    _, fp, _, tn = confusion(field, truth, threshold)
    return fp / (fp + tn) if fp + tn else NAN


def f1_score(
    field: cpt.NDArray, truth: cpt.NDArray, threshold: float | None = None
) -> float:
    """Harmonic mean of `precision` and `recall`, `2 tp / (2 tp + fp + fn)`"""
    tp, fp, fn, _ = confusion(field, truth, threshold)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0 if fp or fn else NAN


def pr_auc(field: cpt.NDArray, truth: cpt.NDArray) -> float:
    """Area under the precision-recall curve, as the average precision."""
    score, label, positives = _ranked(field, truth)
    if positives == 0:
        return NAN
    tp = cp.cumsum(label, dtype=cp.float64)
    counted = cp.arange(1, label.size + 1, dtype=cp.float64)
    # only the last entry of a run of equal scores is a threshold of its own
    ends = cp.append(cp.diff(score) != 0, True)
    tp, counted = tp[ends], counted[ends]
    recalled = tp / positives
    previous = cp.concatenate((cp.zeros(1, dtype=cp.float64), recalled[:-1]))
    return float(cp.sum((recalled - previous) * (tp / counted)))


def roc_auc(field: cpt.NDArray, truth: cpt.NDArray) -> float:
    """Area under the receiver operating characteristic, `recall` against `false_positive_rate`."""
    field, truth = _flatten_pair(field, truth)
    label = truth < _resolve_threshold(truth)
    positives = int(cp.count_nonzero(label))
    negatives = label.size - positives
    if positives == 0 or negatives == 0:
        return NAN
    ordered = cp.sort(-field)
    damaged = -field[label]
    left = cp.searchsorted(ordered, damaged, side="left")
    right = cp.searchsorted(ordered, damaged, side="right")
    ranks = 0.5 * float(cp.sum(left + right + 1))  # 1-based mid-ranks
    return (ranks - 0.5 * positives * (positives + 1)) / (positives * negatives)


def non_discreteness(field: cpt.NDArray, region: cpt.NDArray | None = None) -> float:
    """Greyness measure `mean(4 x (1 - x))` (Sigmund 2007), 0 for a design already 0/1.

    Takes no truth, so it also scores a topology optimization result, where the number
    that matters is how far the grey design the optimizer saw is from the thresholded
    one that gets built. See https://doi.org/10.1007/s00158-006-0087-x
    """
    x = cp.asarray(field)
    x = x if region is None else x[region]
    return float(cp.mean(4.0 * x * (1.0 - x))) if x.size else NAN


def l2_error(field: cpt.NDArray, truth: cpt.NDArray, relative: bool = True) -> float:
    """L2 norm of the reconstruction error, normalized by the norm of `truth`."""
    field, truth = _flatten_pair(field, truth)
    error = float(cp.linalg.norm(field - truth))
    return error / float(cp.linalg.norm(truth)) if relative else error
