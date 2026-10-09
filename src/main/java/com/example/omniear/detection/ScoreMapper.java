package com.example.omniear.detection;

import com.example.omniear.model.ModelPrediction;
import java.util.EnumMap;
import java.util.Map;

/**
 * Maps model labels to application categories. A category score is the maximum
 * weighted probability across its mapped labels; the driving label is recorded.
 */
public final class ScoreMapper {

    private final DetectionConfig config;

    public ScoreMapper(DetectionConfig config) {
        this.config = config;
    }

    public MappedScores map(ModelPrediction prediction) {
        Map<ThreatCategory, Float> scores = new EnumMap<>(ThreatCategory.class);
        Map<ThreatCategory, String> labels = new EnumMap<>(ThreatCategory.class);
        for (CategoryRule rule : config.categories) {
            float best = 0f;
            String bestLabel = null;
            for (Map.Entry<String, Float> e : rule.labels().entrySet()) {
                float weighted = e.getValue() * prediction.score(e.getKey());
                if (weighted > best) {
                    best = weighted;
                    bestLabel = e.getKey();
                }
            }
            if (bestLabel != null) {
                scores.put(rule.category(), best);
                labels.put(rule.category(), bestLabel);
            }
        }
        return new MappedScores(scores, labels);
    }
}
