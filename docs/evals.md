# Evals

**Evals** scores a reconstructed design field against the truth, both as a detection problem (is each node damaged or intact) and as a field error

Every metric takes the pair `(field, truth)` over the same grid and flattens it, so the caller decides beforehand which nodes are compared: the drivers slice off the ghost ring first. The truth is split at the midpoint of its own range, which is the natural cut for a two-material indicator, while `field` is split at `threshold` where one is given and at the same midpoint otherwise

| member | signature | description |
|---|---|---|
| counts | `confusion(field, truth, threshold=None)` | the `(tp, fp, fn, tn)` every rate below is built from, damage being the positive class |
| rate | `precision(field, truth, threshold=None)` | share of the detected damage that is real, $tp/(tp+fp)$ |
| rate | `recall(field, truth, threshold=None)` | share of the real damage that is detected, $tp/(tp+fn)$ |
| rate | `false_positive_rate(field, truth, threshold=None)` | share of the intact material flagged as damage, $fp/(fp+tn)$ |
| rate | `f1_score(field, truth, threshold=None)` | harmonic mean of the two, $2tp/(2tp+fp+fn)$ |
| threshold-free | `pr_auc(field, truth)` | area under the precision–recall curve, as the average precision |
| threshold-free | `roc_auc(field, truth)` | area under `recall` against `false_positive_rate` |
| greyness | `non_discreteness(field, region=None)` | $\overline{4x(1-x)}$ over `region`, 0 once the design is already 0/1, and the only metric here that needs no truth |
| field error | `l2_error(field, truth, relative=True)` | $\lVert x-x_\textrm{truth}\rVert_2$, normalized by $\lVert x_\textrm{truth}\rVert_2$ unless `relative` is false |

The two areas answer different questions and the choice is the class balance. `roc_auc` weighs the two classes symmetrically, so it stays informative when damage occupies a sizeable share of the grid, but a handful of voids among tens of thousands of intact nodes leaves it near 1 for almost any reconstruction: the intact class dominates both of its axes. `pr_auc` never looks at true negatives, so it keeps discriminating exactly there, which is the regime an inversion driver is normally in

`non_discreteness` scores a design against itself rather than against a truth, which is what a [tato](tato.md) run has: no reference structure exists, only the grey field the optimizer saw and the thresholded one that gets built. It is the number that explains the gap between their two objectives, since a design already near 0/1 cannot lose much to being snapped ([Sigmund 2007](https://doi.org/10.1007/s00158-006-0087-x))

A rate whose denominator is empty returns `NAN` rather than 0, so a run that detected nothing is visibly different from one that detected nothing correctly. The threshold-free pair needs no `threshold` at all, which makes them the honest comparison between two reconstructions that would each want a different cut
