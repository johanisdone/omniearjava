package com.example.omniear.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Immutable model output: ordered labels and their scores. */
public record ModelPrediction(List<String> labels, float[] scores) {

    public ModelPrediction {
        if (labels.size() != scores.length) {
            throw new IllegalArgumentException(
                    "labels (" + labels.size() + ") != scores (" + scores.length + ")");
        }
    }

    public int size() {
        return scores.length;
    }

    public float score(String label) {
        int i = labels.indexOf(label);
        return i < 0 ? 0f : scores[i];
    }

    public int argmax() {
        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        return best;
    }

    public Map<String, Float> asMap() {
        Map<String, Float> m = new LinkedHashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            m.put(labels.get(i), scores[i]);
        }
        return m;
    }

    public List<Map.Entry<String, Float>> topK(int k) {
        List<Map.Entry<String, Float>> entries = new ArrayList<>(asMap().entrySet());
        entries.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
        return entries.subList(0, Math.min(k, entries.size()));
    }
}
