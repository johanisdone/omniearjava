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
 * <p>For each category an <em>optional</em> score transform (trailing moving
 * average, moving maximum, or EMA) smooths the per-window score first; the
 * transformed score must then exceed its threshold in at least
 * {@code confirmOf} of the last {@code confirmWindows} windows before it fires.
 * With {@code confirmOf == confirmWindows} this is the strict "N consecutive
 * windows" hysteresis, and {@code 2-of-3} tolerates a single gap. A cooldown
 * then suppresses repeated alerts during one continuous event. The peak raw
 * score and first/last timestamps are preserved. Honestly ambiguous windows
 * are logged rather than silently labelled {@code NORMAL}.
 */
public final class TemporalDecisionEngine {

    private static final Logger LOG = LoggerFactory.getLogger(TemporalDecisionEngine.class);

    private static final class State {
        /** Confirmation bits for the last {@code confirmWindows} windows. */
        final boolean[] above;
        /** Raw score of those windows (peak/timestamp reporting). */
        final float[] score;
        final long[] time;
        /** Raw score history for trailing transforms; slot {@code raw.length - 1} is newest. */
        final float[] raw;
        /** How many trailing raw slots currently hold a real (non-initial) value. */
        int filled;
        /** EMA accumulator. */
        float ema;
        long lastFiredMs = Long.MIN_VALUE;

        State(int confirmWindows, int rawLength) {
            this.above = new boolean[confirmWindows];
            this.score = new float[confirmWindows];
            this.time = new long[confirmWindows];
            this.raw = new float[Math.max(1, rawLength)];
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
            states.put(rule.category(), new State(rule.confirmWindows(), rule.transformSpan()));
        }
    }

    public List<ThreatEvent> update(long timestampMs, MappedScores mapped) {
        return update(timestampMs, mapped, Float.NaN);
    }

    /**
     * As {@link #update(long, MappedScores)} but with the window's RMS level in
     * dBFS. Windows quieter than {@link DetectionConfig#noiseGateDb} are treated
     * as background and cannot raise an alert; pass {@link Float#NaN} to disable
     * the gate for this window.
     */
    public List<ThreatEvent> update(long timestampMs, MappedScores mapped, float rmsDb) {
        if (!Float.isNaN(rmsDb) && rmsDb < config.noiseGateDb) {
            LOG.debug("gated background window t={}ms rms={}dB", timestampMs,
                    String.format("%.1f", rmsDb));
            return List.of();
        }
        List<ThreatEvent> events = new ArrayList<>();
        long nowWall = System.currentTimeMillis();

        for (CategoryRule rule : config.categories) {
            ThreatCategory cat = rule.category();
            State st = states.get(cat);
            float raw = mapped.score(cat);

            float value = applyTransform(st, rule, raw);

            // Slide the confirmation window one step and append the current result.
            int last = st.above.length - 1;
            for (int i = 0; i < last; i++) {
                st.above[i] = st.above[i + 1];
                st.score[i] = st.score[i + 1];
                st.time[i] = st.time[i + 1];
            }
            boolean hit = value >= rule.threshold();
            st.above[last] = hit;
            st.score[last] = raw;
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

    /**
     * Pushes the raw score into the transform history and returns the smoothed
     * value used for thresholding. Mirrors {@code decision_lab.transform()}:
     * trailing windows average/max over the entries actually seen so far, and
     * the EMA starts from 0.
     */
    private float applyTransform(State st, CategoryRule rule, float raw) {
        // Shift raw history and append.
        int rl = st.raw.length;
        for (int i = 0; i < rl - 1; i++) {
            st.raw[i] = st.raw[i + 1];
        }
        st.raw[rl - 1] = raw;
        if (st.filled < rl) {
            st.filled++;
        }

        return switch (rule.transform()) {
            case NONE -> raw;
            case MOVAVG -> {
                int n = Math.min(rule.transformSpan(), st.filled);
                float sum = 0f;
                for (int i = rl - n; i < rl; i++) {
                    sum += st.raw[i];
                }
                yield n == 0 ? raw : sum / n;
            }
            case MOVMAX -> {
                int n = Math.min(rule.transformSpan(), st.filled);
                float max = Float.NEGATIVE_INFINITY;
                for (int i = rl - n; i < rl; i++) {
                    if (st.raw[i] > max) {
                        max = st.raw[i];
                    }
                }
                yield n == 0 ? raw : max;
            }
            case EMA -> {
                st.ema = rule.param() * raw + (1f - rule.param()) * st.ema;
                yield st.ema;
            }
        };
    }
}
