package com.example.omniear.eval;

import com.example.omniear.detection.ThreatCategory;

/** Precision/recall plus detection-delay statistics for one category. */
public final class CategoryMetrics {

    public final ThreatCategory category;
    public final int tp;
    public final int fp;
    public final int fn;
    public final double precision;
    public final double recall;
    public final double f1;
    public long meanDelayMs = -1;
    public long medianDelayMs = -1;
    public long maxDelayMs = -1;

    public CategoryMetrics(ThreatCategory category, int tp, int fp, int fn) {
        this.category = category;
        this.tp = tp;
        this.fp = fp;
        this.fn = fn;
        this.precision = tp + fp == 0 ? 0 : tp / (double) (tp + fp);
        this.recall = tp + fn == 0 ? 0 : tp / (double) (tp + fn);
        this.f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    }
}
