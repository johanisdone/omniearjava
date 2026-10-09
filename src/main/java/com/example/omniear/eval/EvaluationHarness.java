package com.example.omniear.eval;

import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.model.AudioPreprocessor;
import com.example.omniear.model.CedMiniThreatModel;
import com.example.omniear.model.ModelConfig;
import com.example.omniear.model.ModelPrediction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs the full pipeline losslessly over a labelled dataset and produces an
 * {@link EvaluationReport}. Unlike live capture, every window is processed (no
 * drop-stale), so results are deterministic and reproducible.
 */
public final class EvaluationHarness {

    private static final Logger LOG = LoggerFactory.getLogger(EvaluationHarness.class);

    private final ModelConfig modelConfig;
    private final DetectionConfig detectionConfig;
    private final Path modelsDir;

    public EvaluationHarness(ModelConfig modelConfig, DetectionConfig detectionConfig, Path modelsDir) {
        this.modelConfig = modelConfig;
        this.detectionConfig = detectionConfig;
        this.modelsDir = modelsDir;
    }

    public EvaluationReport run(Path datasetDir) throws Exception {
        List<EvalCase> cases = EvalDataset.load(datasetDir.resolve("labels.csv"));
        int win = modelConfig.windowSamples();
        int hop = modelConfig.hopSamples();
        List<ClipResult> results = new ArrayList<>();
        long preprocessNanos = 0;
        long inferenceNanos = 0;
        long decisionNanos = 0;
        long windowsProcessed = 0;

        try (CedMiniThreatModel model = CedMiniThreatModel.fromModelsDir(modelsDir, modelConfig)) {
            for (EvalCase c : cases) {
                if (!Files.isRegularFile(c.file())) {
                    LOG.warn("skipping missing file {}", c.file());
                    continue;
                }
                long t0 = System.nanoTime();
                float[] samples = AudioPreprocessor.readMono(c.file(), modelConfig.sr);
                int[] offsets = AudioPreprocessor.windowOffsets(samples.length, win, hop);
                long durationMs = samples.length * 1000L / modelConfig.sr;
                preprocessNanos += System.nanoTime() - t0;

                ScoreMapper mapper = new ScoreMapper(detectionConfig);
                TemporalDecisionEngine engine = new TemporalDecisionEngine(detectionConfig);
                List<ThreatEvent> events = new ArrayList<>();
                for (int o : offsets) {
                    long t1 = System.nanoTime();
                    float[] window = AudioPreprocessor.window(samples, o, win);
                    long t2 = System.nanoTime();
                    ModelPrediction prediction = model.predict(window);
                    long t3 = System.nanoTime();
                    long endMs = (o + win) * 1000L / modelConfig.sr;
                    events.addAll(engine.update(endMs, mapper.map(prediction)));
                    long t4 = System.nanoTime();
                    preprocessNanos += t2 - t1;
                    inferenceNanos += t3 - t2;
                    decisionNanos += t4 - t3;
                    windowsProcessed++;
                }
                results.add(new ClipResult(c.file(), c.label(), c.onsetMs(), durationMs, events));
                LOG.info("{} label={} alerts={}", c.file().getFileName(), c.label(), events.size());
            }
        }
        EvaluationReport report = EvaluationReport.compute(results, modelConfig, detectionConfig);
        if (windowsProcessed > 0) {
            report.preprocessMsPerWindow = preprocessNanos / 1_000_000.0 / windowsProcessed;
            report.inferenceMsPerWindow = inferenceNanos / 1_000_000.0 / windowsProcessed;
            report.decisionMsPerWindow = decisionNanos / 1_000_000.0 / windowsProcessed;
        }
        return report;
    }
}
