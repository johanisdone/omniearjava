package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.example.omniear.detection.CategoryRule;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.eval.CategoryMetrics;
import com.example.omniear.eval.ClipResult;
import com.example.omniear.eval.EvaluationReport;
import com.example.omniear.model.ModelConfig;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class EvaluationMetricsTest {

    private static final Path SIREN_POS = Path.of("siren_pos.wav");
    private static final Path SIREN_POS_MISS = Path.of("siren_miss.wav");
    private static final Path BG_FP = Path.of("background_fp.wav");
    private static final Path BG = Path.of("background.wav");

    private static ThreatEvent siren(long startMs) {
        return new ThreatEvent(ThreatCategory.SIREN, 0.9f, startMs, startMs + 1000, "siren", 0L);
    }

    @Test
    void computesPrecisionRecallDelayAndFalseAlertsPerHour() {
        List<ClipResult> results = List.of(
                new ClipResult(SIREN_POS, ThreatCategory.SIREN, 0, 5000, List.of(siren(2000))),
                new ClipResult(SIREN_POS_MISS, ThreatCategory.SIREN, 0, 5000, List.of()),
                new ClipResult(BG_FP, ThreatCategory.NORMAL, -1, 10000, List.of(siren(3000))),
                new ClipResult(BG, ThreatCategory.NORMAL, -1, 10000, List.of()));

        ModelConfig modelConfig = new ModelConfig();
        modelConfig.sr = 16000;
        modelConfig.win_s = 2.0;
        modelConfig.hop_s = 1.0;

        DetectionConfig cfg = new DetectionConfig();
        cfg.uncertainFloor = 0.3f;
        cfg.categories = List.of(
                new CategoryRule(ThreatCategory.SIREN, Map.of("siren", 1.0f), 0.6f, 2, 10000),
                new CategoryRule(ThreatCategory.GUNSHOT, Map.of("gunshot", 1.0f), 0.9f, 2, 10000));

        EvaluationReport report = EvaluationReport.compute(results, modelConfig, cfg);

        assertEquals(4, report.clips);
        assertEquals(2, report.positiveClips);
        assertEquals(2, report.negativeClips);
        assertEquals(1, report.totalFalseAlerts);
        // 1 false alert over 20000 ms = 0.005555 h -> 180 per hour
        assertEquals(180.0, report.falseAlertsPerHour, 1e-6);

        CategoryMetrics siren = report.perCategory.stream()
                .filter(m -> m.category == ThreatCategory.SIREN).findFirst().orElseThrow();
        assertEquals(1, siren.tp);
        assertEquals(1, siren.fp);
        assertEquals(1, siren.fn);
        assertEquals(0.5, siren.precision, 1e-9);
        assertEquals(0.5, siren.recall, 1e-9);
        assertEquals(0.5, siren.f1, 1e-9);
        assertEquals(2000, siren.meanDelayMs);
        assertEquals(2000, siren.maxDelayMs);

        assertEquals(0.5, report.macroF1, 1e-9);
    }
}
