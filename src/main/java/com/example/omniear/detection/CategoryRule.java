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
 */
public record CategoryRule(
        ThreatCategory category,
        Map<String, Float> labels,
        float threshold,
        int confirmWindows,
        int confirmOf,
        long cooldownMs) {

    /** Convenience constructor requiring every window in the span (strictly consecutive). */
    public CategoryRule(ThreatCategory category,
                        Map<String, Float> labels,
                        float threshold,
                        int confirmWindows,
                        long cooldownMs) {
        this(category, labels, threshold, confirmWindows, confirmWindows, cooldownMs);
    }

    public CategoryRule {
        if (confirmWindows < 1) {
            confirmWindows = 1;
        }
        if (confirmOf < 1 || confirmOf > confirmWindows) {
            confirmOf = confirmWindows;
        }
    }
}
