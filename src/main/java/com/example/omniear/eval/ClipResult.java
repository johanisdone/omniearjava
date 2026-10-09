package com.example.omniear.eval;

import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import java.nio.file.Path;
import java.util.List;

/** Outcome of running the pipeline over one labelled clip. */
public record ClipResult(
        Path file,
        ThreatCategory label,
        long onsetMs,
        long durationMs,
        List<ThreatEvent> events) {

    public boolean fired(ThreatCategory category) {
        return events.stream().anyMatch(e -> e.category() == category);
    }

    /** First event of {@code category} in audio time, or null. */
    public ThreatEvent first(ThreatCategory category) {
        return events.stream().filter(e -> e.category() == category).findFirst().orElse(null);
    }

    public long countAlerts() {
        return events.size();
    }
}
