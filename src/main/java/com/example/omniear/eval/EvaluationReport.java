package com.example.omniear.eval;

import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.model.ModelConfig;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Aggregated evaluation results: per-category precision/recall/F1, detection
 * delay, and false alerts per hour. Serialisable to JSON via Gson.
 */
public final class EvaluationReport {

    public String generatedAt;
    public String javaVersion;
    public int sampleRate;
    public double windowSeconds;
    public double hopSeconds;

    public int clips;
    public int positiveClips;
    public int negativeClips;
    public long negativeDurationMs;

    public int totalFalseAlerts;
    public double falseAlertsPerHour;

    public double macroF1;

    public double preprocessMsPerWindow;
    public double inferenceMsPerWindow;
    public double decisionMsPerWindow;

    public List<CategoryMetrics> perCategory = new ArrayList<>();

    private EvaluationReport() {
    }

    public static EvaluationReport compute(List<ClipResult> results,
                                           ModelConfig modelConfig,
                                           DetectionConfig detectionConfig) {
        EvaluationReport r = new EvaluationReport();
        r.generatedAt = Instant.now().toString();
        r.javaVersion = System.getProperty("java.version");
        r.sampleRate = modelConfig.sr;
        r.windowSeconds = modelConfig.win_s;
        r.hopSeconds = modelConfig.hop_s;
        r.clips = results.size();

        long negMs = 0;
        int falseAlerts = 0;
        for (ClipResult c : results) {
            if (c.label() == ThreatCategory.NORMAL) {
                r.negativeClips++;
                negMs += c.durationMs();
                falseAlerts += c.countAlerts();
            } else {
                r.positiveClips++;
            }
        }
        r.negativeDurationMs = negMs;
        r.totalFalseAlerts = falseAlerts;
        double negHours = negMs / 3_600_000.0;
        r.falseAlertsPerHour = negHours > 0 ? falseAlerts / negHours : 0;

        double f1Sum = 0;
        int counted = 0;
        for (var rule : detectionConfig.categories) {
            CategoryMetrics m = perCategory(results, rule.category());
            r.perCategory.add(m);
            if (m.tp + m.fp + m.fn > 0) {
                f1Sum += m.f1;
                counted++;
            }
        }
        r.macroF1 = counted > 0 ? f1Sum / counted : 0;
        return r;
    }

    private static CategoryMetrics perCategory(List<ClipResult> results, ThreatCategory cat) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        List<Long> delays = new ArrayList<>();
        for (ClipResult c : results) {
            boolean fired = c.fired(cat);
            if (c.label() == cat) {
                if (fired) {
                    tp++;
                    ThreatEvent e = c.first(cat);
                    if (c.onsetMs() >= 0 && e != null) {
                        delays.add(Math.max(0, e.startMs() - c.onsetMs()));
                    }
                } else {
                    fn++;
                }
            } else if (fired) {
                fp++;
            }
        }
        CategoryMetrics m = new CategoryMetrics(cat, tp, fp, fn);
        if (!delays.isEmpty()) {
            Collections.sort(delays);
            long sum = 0;
            for (long d : delays) {
                sum += d;
            }
            m.meanDelayMs = sum / delays.size();
            m.medianDelayMs = delays.get(delays.size() / 2);
            m.maxDelayMs = delays.get(delays.size() - 1);
        }
        return m;
    }

    public void print() {
        System.out.println();
        System.out.println("=== OmniEar evaluation ===");
        System.out.printf("clips=%d (positives=%d, negatives=%d)  negative audio=%.2f h%n",
                clips, positiveClips, negativeClips, negativeDurationMs / 3_600_000.0);
        System.out.printf("false alerts/hour (on NORMAL clips): %.2f  (total=%d)  macroF1=%.3f%n",
                falseAlertsPerHour, totalFalseAlerts, macroF1);
        System.out.printf("per-window ms: preprocess=%.2f  inference=%.2f  decision=%.2f%n",
                preprocessMsPerWindow, inferenceMsPerWindow, decisionMsPerWindow);
        System.out.println();
        System.out.printf("%-16s %4s %4s %4s %6s %6s %6s %9s %9s%n",
                "category", "tp", "fp", "fn", "prec", "rec", "f1", "meanDly", "maxDly");
        for (CategoryMetrics m : perCategory) {
            System.out.printf("%-16s %4d %4d %4d %6.2f %6.2f %6.2f %9s %9s%n",
                    m.category, m.tp, m.fp, m.fn, m.precision, m.recall, m.f1,
                    m.meanDelayMs < 0 ? "-" : m.meanDelayMs + "ms",
                    m.maxDelayMs < 0 ? "-" : m.maxDelayMs + "ms");
        }
    }

    public String toJson() {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(this);
    }
}
