package com.example.omniear.detection;

import java.util.EnumMap;
import java.util.Map;

/**
 * Per-category scores produced by {@link ScoreMapper}: the mapped category score
 * and the model label that drove it (for explainability).
 */
public record MappedScores(Map<ThreatCategory, Float> scores,
                           Map<ThreatCategory, String> drivingLabels) {

    public float score(ThreatCategory category) {
        return scores.getOrDefault(category, 0f);
    }

    /** Returns a copy with {@code category} raised to at least {@code value}. */
    public MappedScores withScore(ThreatCategory category, float value, String label) {
        Map<ThreatCategory, Float> raisedScores = new EnumMap<>(scores);
        Map<ThreatCategory, String> raisedLabels = new EnumMap<>(drivingLabels);
        raisedScores.put(category, Math.max(raisedScores.getOrDefault(category, 0f), value));
        raisedLabels.put(category, label);
        return new MappedScores(raisedScores, raisedLabels);
    }

    public ThreatCategory strongest() {
        ThreatCategory best = ThreatCategory.NORMAL;
        float bestScore = -1f;
        for (Map.Entry<ThreatCategory, Float> e : scores.entrySet()) {
            if (e.getValue() > bestScore) {
                bestScore = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    public float strongestScore() {
        float bestScore = 0f;
        for (float v : scores.values()) {
            bestScore = Math.max(bestScore, v);
        }
        return bestScore;
    }
}
