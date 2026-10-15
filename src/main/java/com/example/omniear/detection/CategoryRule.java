package com.example.omniear.detection;

import java.util.Map;

/**
 * Decision rule for one application category.
 *
 * @param category       application category
 * @param labels         model label -> weight (category score is the max weighted prob)
 * @param threshold      category score threshold
 * @param confirmWindows size of the sliding confirmation window
 * @param confirmOf      how many of those windows must exceed the threshold to fire;
 *                       defaults to {@code confirmWindows} (strictly consecutive)
 * @param cooldownMs     minimum time between emitted alerts for this category
 * @param transform      optional score smoothing applied before thresholding
 * @param param          transform smoothing parameter (window span or EMA factor)
 */
public record CategoryRule(
        ThreatCategory category,
        Map<String, Float> labels,
        float threshold,
        int confirmWindows,
        int confirmOf,
        long cooldownMs,
        TransformKind transform,
        float param) {

    /** Convenience constructor requiring every window in the span (strictly consecutive). */
    public CategoryRule(ThreatCategory category,
                        Map<String, Float> labels,
                        float threshold,
                        int confirmWindows,
                        long cooldownMs) {
        this(category, labels, threshold, confirmWindows, confirmWindows, cooldownMs);
    }

    /** Convenience constructor with {@code confirmOf} but no score transform. */
    public CategoryRule(ThreatCategory category,
                        Map<String, Float> labels,
                        float threshold,
                        int confirmWindows,
                        int confirmOf,
                        long cooldownMs) {
        this(category, labels, threshold, confirmWindows, confirmOf, cooldownMs, TransformKind.NONE, 0f);
    }

    public CategoryRule {
        if (confirmWindows < 1) {
            confirmWindows = 1;
        }
        if (confirmOf < 1 || confirmOf > confirmWindows) {
            confirmOf = confirmWindows;
        }
        if (transform == null) {
            transform = TransformKind.NONE;
        }
        if (param <= 0f) {
            param = transform.defaultParam();
        }
    }

    /**
     * Number of raw windows the trailing transform looks back over
     * (1 for {@link TransformKind#NONE} and for EMA, which has no fixed span).
     */
    public int transformSpan() {
        return switch (transform) {
            case MOVAVG, MOVMAX -> Math.max(1, Math.round(param));
            default -> 1;
        };
    }
}
