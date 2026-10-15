package com.example.omniear.detection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Explainable temporal logic over per-window category scores.
 *
 * <p>For each category a score must exceed its threshold in at least
 * {@code confirmOf} of the last {@code confirmWindows} windows before it fires;
 * with {@code confirmOf == confirmWindows} this is the strict "N consecutive
 * windows" hysteresis, and {@code 2-of-3} tolerates a single gap. A cooldown
 * then suppresses repeated alerts during one continuous event. The peak score
 * and first/last timestamps are preserved. Honestly ambiguous windows are logged
 * rather than silently labelled {@code NORMAL}.
 */
public final class TemporalDecisionEngine {

    private static final Logger LOG = LoggerFactory.getLogger(TemporalDecisionEngine.class);

    private static final class State {
        final boolean[] above;
        final float[] score;
        final long[] time;
        long lastFiredMs = Long.MIN_VALUE;

        State(int size) {
            this.above = new boolean[size];
            this.score = new float[size];
            this.time = new long[size];
        }

        void clear() {
            Arrays.fill(above, false);
        }
    }

    private final DetectionConfig config;
    private final Map<ThreatCategory, State> states = new EnumMap<>(ThreatCategory.class);

    public TemporalDecisionEngine(DetectionConfig config) {
        this.config = config;
        for (CategoryRule rule : config.categories) {
            states.put(rule.category(), new State(rule.confirmWindows()));
        }
    }

    public List<ThreatEvent> update(long timestampMs, MappedScores mapped) {
        List<ThreatEvent> events = new ArrayList<>();
        long nowWall = System.currentTimeMillis();

        for (CategoryRule rule : config.categories) {
            ThreatCategory cat = rule.category();
            State st = states.get(cat);
            float score = mapped.score(cat);

            // Slide the confirmation window one step and append the current result.
            int last = st.above.length - 1;
            for (int i = 0; i < last; i++) {
                st.above[i] = st.above[i + 1];
                st.score[i] = st.score[i + 1];
                st.time[i] = st.time[i + 1];
            }
            boolean hit = score >= rule.threshold();
            st.above[last] = hit;
            st.score[last] = score;
            st.time[last] = timestampMs;

            int hits = 0;
            float peak = 0f;
            long firstMs = Long.MAX_VALUE;
            for (int i = 0; i < st.above.length; i++) {
                if (st.above[i]) {
                    hits++;
                    if (st.score[i] > peak) {
                        peak = st.score[i];
                    }
                    if (st.time[i] < firstMs) {
                        firstMs = st.time[i];
                    }
                }
            }

            boolean confirmed = hits >= rule.confirmOf();
            boolean coolingDown = st.lastFiredMs != Long.MIN_VALUE
                    && timestampMs - st.lastFiredMs < rule.cooldownMs();
            if (confirmed && !coolingDown) {
                String label = mapped.drivingLabels().getOrDefault(cat, cat.name());
                events.add(new ThreatEvent(cat, peak, firstMs, timestampMs, label, nowWall));
                st.lastFiredMs = timestampMs;
                st.clear();
            }
        }

        if (events.isEmpty() && mapped.strongestScore() >= config.uncertainFloor) {
            LOG.debug("ambiguous window t={}ms strongest={} score={}",
                    timestampMs, mapped.strongest(), String.format("%.3f", mapped.strongestScore()));
        }
        return events;
    }
}
