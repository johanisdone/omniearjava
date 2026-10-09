package com.example.omniear.detection;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Explainable temporal logic over per-window category scores.
 *
 * <p>For each category: a score must exceed its threshold for
 * {@code confirmWindows} consecutive windows (the two-window hysteresis),
 * then a cooldown suppresses repeated alerts during one continuous event.
 * The peak score and first/last timestamps are preserved. Honestly ambiguous
 * windows are logged rather than silently labelled {@code NORMAL}.
 */
public final class TemporalDecisionEngine {

    private static final Logger LOG = LoggerFactory.getLogger(TemporalDecisionEngine.class);

    private static final class State {
        int consecutive;
        long firstAboveMs = Long.MIN_VALUE;
        float peak = 0f;
        long lastFiredMs = Long.MIN_VALUE;
    }

    private final DetectionConfig config;
    private final Map<ThreatCategory, State> states = new EnumMap<>(ThreatCategory.class);

    public TemporalDecisionEngine(DetectionConfig config) {
        this.config = config;
        for (CategoryRule rule : config.categories) {
            states.put(rule.category(), new State());
        }
    }

    public List<ThreatEvent> update(long timestampMs, MappedScores mapped) {
        List<ThreatEvent> events = new ArrayList<>();
        long nowWall = System.currentTimeMillis();

        for (CategoryRule rule : config.categories) {
            ThreatCategory cat = rule.category();
            State st = states.get(cat);
            float score = mapped.score(cat);

            if (score >= rule.threshold()) {
                if (st.consecutive == 0) {
                    st.firstAboveMs = timestampMs;
                    st.peak = score;
                }
                st.consecutive++;
                st.peak = Math.max(st.peak, score);

                boolean confirmed = st.consecutive >= rule.confirmWindows();
                boolean coolingDown = st.lastFiredMs != Long.MIN_VALUE
                        && timestampMs - st.lastFiredMs < rule.cooldownMs();
                if (confirmed && !coolingDown) {
                    String label = mapped.drivingLabels().getOrDefault(cat, cat.name());
                    events.add(new ThreatEvent(cat, st.peak, st.firstAboveMs, timestampMs, label, nowWall));
                    st.lastFiredMs = timestampMs;
                    st.consecutive = 0;
                    st.peak = 0f;
                    st.firstAboveMs = Long.MIN_VALUE;
                }
            } else {
                st.consecutive = 0;
                st.peak = 0f;
                st.firstAboveMs = Long.MIN_VALUE;
            }
        }

        if (events.isEmpty() && mapped.strongestScore() >= config.uncertainFloor) {
            LOG.debug("ambiguous window t={}ms strongest={} score={}",
                    timestampMs, mapped.strongest(), String.format("%.3f", mapped.strongestScore()));
        }
        return events;
    }
}
