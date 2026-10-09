package com.example.omniear.detection;

/**
 * An emitted threat event.
 *
 * @param category      application category
 * @param score         peak mapped score that caused the event
 * @param startMs       audio-time of the first window above threshold (ms)
 * @param endMs         audio-time of the confirming window (ms)
 * @param drivingLabel  model label that produced the score
 * @param emittedWallMs wall-clock time the event was emitted (ms)
 */
public record ThreatEvent(
        ThreatCategory category,
        float score,
        long startMs,
        long endMs,
        String drivingLabel,
        long emittedWallMs) {

    public long durationMs() {
        return endMs - startMs;
    }
}
