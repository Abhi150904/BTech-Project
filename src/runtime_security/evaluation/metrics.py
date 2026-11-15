"""Metrics use JSON null when a rate's denominator is zero."""

from collections.abc import Sequence


def rate(n, d):
    return n / d if d else None


def confusion(truth: Sequence[bool | None], predicted: Sequence[bool]) -> dict:
    pairs = list(zip(truth, predicted, strict=True))
    tp = sum(y is True and p for y, p in pairs)
    fp = sum(y is False and p for y, p in pairs)
    fn = sum(y is True and not p for y, p in pairs)
    tn = sum(y is False and not p for y, p in pairs)
    return dict(
        n=len(pairs),
        positive=tp + fn,
        negative=tn + fp,
        unknown=len(pairs) - tp - fp - fn - tn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        fpr=rate(fp, fp + tn),
        precision=rate(tp, tp + fp),
        recall=rate(tp, tp + fn),
        f1=rate(2 * tp, 2 * tp + fp + fn),
    )


def calibrate(truth: Sequence[bool | None], scores: Sequence[float]) -> dict:
    known = [(y, s) for y, s in zip(truth, scores, strict=True) if y is not None]
    curve = []
    for t in sorted({0.0, 0.5, 1.0, *(s for _, s in known)}):
        m = confusion(truth, [s >= t for s in scores])
        curve.append({"threshold": t, **m})
    if {y for y, _ in known} != {False, True}:
        return dict(threshold=0.5, method="fixed_fallback_missing_validation_class", curve=curve)
    best = max(curve, key=lambda m: (m["f1"] or 0, -m["fp"], m["threshold"]))
    return dict(
        threshold=best["threshold"],
        method="validation_f1_then_fewer_fp_then_higher_threshold",
        curve=curve,
    )
