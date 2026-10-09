package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.example.omniear.detection.CategoryRule;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.MappedScores;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.model.ModelPrediction;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DetectionLogicTest {

    private static final List<String> LABELS =
            List.of("scream", "gunshot", "vehicle_crash", "knock", "siren");

    private static ModelPrediction prediction(float scream, float gunshot,
                                              float crash, float knock, float siren) {
        return new ModelPrediction(LABELS, new float[]{scream, gunshot, crash, knock, siren});
    }

    private static DetectionConfig config(CategoryRule... rules) {
        DetectionConfig cfg = new DetectionConfig();
        cfg.categories = List.of(rules);
        cfg.uncertainFloor = 0.3f;
        return cfg;
    }

    @Test
    void mapperUsesWeightedMaxAndDrivingLabel() {
        DetectionConfig cfg = config(
                new CategoryRule(ThreatCategory.SCREAM_DISTRESS, Map.of("scream", 1.0f), 0.9f, 2, 1000),
                new CategoryRule(ThreatCategory.IMPACT_CRASH, Map.of("vehicle_crash", 1.0f, "knock", 0.5f), 0.9f, 2, 1000));
        ScoreMapper mapper = new ScoreMapper(cfg);

        MappedScores m = mapper.map(prediction(0.95f, 0f, 0.5f, 0.9f, 0f));
        assertEquals(0.95f, m.score(ThreatCategory.SCREAM_DISTRESS), 1e-6);
        assertEquals("scream", m.drivingLabels().get(ThreatCategory.SCREAM_DISTRESS));
        // vehicle_crash 0.5 vs knock 0.9*0.5=0.45 -> 0.5
        assertEquals(0.5f, m.score(ThreatCategory.IMPACT_CRASH), 1e-6);
        assertEquals("vehicle_crash", m.drivingLabels().get(ThreatCategory.IMPACT_CRASH));
    }

    @Test
    void requiresConsecutiveWindowsThenCooldown() {
        DetectionConfig cfg = config(
                new CategoryRule(ThreatCategory.GUNSHOT, Map.of("gunshot", 1.0f), 0.5f, 2, 5000));
        TemporalDecisionEngine engine = new TemporalDecisionEngine(cfg);
        MappedScores hot = mapped(ThreatCategory.GUNSHOT, 0.6f);

        assertTrue(engine.update(0, hot).isEmpty(), "one window is not enough");
        assertEquals(1, engine.update(1000, hot).size(), "second consecutive window fires");

        // After firing at 1000ms, a continuous run must not re-alert until the
        // 5000 ms cooldown elapses (i.e. not again before 6000 ms).
        assertTrue(engine.update(2000, hot).isEmpty());
        assertTrue(engine.update(3000, hot).isEmpty(), "cooldown suppresses repeat");
        assertTrue(engine.update(5000, hot).isEmpty(), "still within cooldown");
        assertEquals(1, engine.update(6000, hot).size(), "cooldown elapsed -> re-alert");
    }

    @Test
    void belowThresholdResetsConsecutiveCount() {
        DetectionConfig cfg = config(
                new CategoryRule(ThreatCategory.SIREN, Map.of("siren", 1.0f), 0.5f, 2, 0));
        TemporalDecisionEngine engine = new TemporalDecisionEngine(cfg);
        MappedScores hot = mapped(ThreatCategory.SIREN, 0.9f);
        MappedScores cold = mapped(ThreatCategory.SIREN, 0.1f);

        assertTrue(engine.update(0, hot).isEmpty());
        assertTrue(engine.update(1000, cold).isEmpty());
        assertTrue(engine.update(2000, hot).isEmpty(), "reset means we need two again");
        assertEquals(1, engine.update(3000, hot).size());
    }

    private static MappedScores mapped(ThreatCategory category, float score) {
        Map<ThreatCategory, Float> scores = new EnumMap<>(ThreatCategory.class);
        Map<ThreatCategory, String> labels = new EnumMap<>(ThreatCategory.class);
        scores.put(category, score);
        labels.put(category, category.name().toLowerCase());
        return new MappedScores(scores, labels);
    }
}
