package com.example.omniear.detection;

/**
 * Optional per-window score transform applied before a category's threshold and
 * confirmation logic.
 *
 * <p>The only transform that measurably helped honest (out-of-fold) evaluation
 * was {@link #MOVAVG} — a trailing moving average. It attenuates isolated,
 * noise-driven spikes (reducing false alerts) while leaving sustained events
 * (which occupy several overlapping windows) intact. {@link #MOVMAX} and
 * {@link #EMA} were evaluated and did not help, but are supported for future
 * tuning.
 */
public enum TransformKind {

    /** Raw per-window score, unchanged. */
    NONE,

    /** Trailing mean of the last {@code param} windows (fewer when history is short). */
    MOVAVG,

    /** Trailing maximum of the last {@code param} windows. */
    MOVMAX,

    /** Exponential moving average with {@code param} as the smoothing factor. */
    EMA;

    /** Case-insensitive parse of a configuration value; blank/unknown means {@link #NONE}. */
    public static TransformKind parse(String value) {
        if (value == null || value.isBlank()) {
            return NONE;
        }
        return switch (value.trim().toLowerCase()) {
            case "movavg" -> MOVAVG;
            case "movmax" -> MOVMAX;
            case "ema" -> EMA;
            default -> NONE;
        };
    }

    /** Default smoothing parameter when configuration omits one. */
    public float defaultParam() {
        return switch (this) {
            case MOVAVG, MOVMAX -> 3f;
            case EMA -> 0.5f;
            case NONE -> 0f;
        };
    }
}
