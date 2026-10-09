package com.example.omniear.detection;

import java.util.Map;

/**
 * Decision rule for one application category.
 *
 * @param category       application category
 * @param labels         model label -> weight (category score is the max weighted prob)
 * @param threshold      category score threshold
 * @param confirmWindows number of consecutive windows that must exceed the threshold
 * @param cooldownMs     minimum time between emitted alerts for this category
 */
public record CategoryRule(
        ThreatCategory category,
        Map<String, Float> labels,
        float threshold,
        int confirmWindows,
        long cooldownMs) {
}
