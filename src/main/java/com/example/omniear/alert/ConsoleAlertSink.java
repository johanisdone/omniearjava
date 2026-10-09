package com.example.omniear.alert;

import com.example.omniear.detection.ThreatEvent;
import java.time.Instant;

/** First-prototype alert output: a structured console line. */
public final class ConsoleAlertSink implements AlertSink {

    @Override
    public void emit(ThreatEvent event) {
        System.out.printf(
                "[ALERT] %s category=%s score=%.3f label=%s start=%dms end=%dms dur=%dms wall=%s%n",
                "THREAT",
                event.category(),
                event.score(),
                event.drivingLabel(),
                event.startMs(),
                event.endMs(),
                event.durationMs(),
                Instant.ofEpochMilli(event.emittedWallMs()));
    }
}
