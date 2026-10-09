package com.example.omniear.app;

import com.example.omniear.alert.AlertSink;
import com.example.omniear.alert.ConsoleAlertSink;
import com.example.omniear.audio.AudioCaptureService;
import com.example.omniear.audio.AudioSource;
import com.example.omniear.audio.AudioWindow;
import com.example.omniear.audio.FileAudioSource;
import com.example.omniear.audio.MicrophoneSource;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.MappedScores;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.eval.EvaluationHarness;
import com.example.omniear.eval.EvaluationReport;
import com.example.omniear.model.CedMiniThreatModel;
import com.example.omniear.model.ModelConfig;
import com.example.omniear.model.ModelPrediction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.LineUnavailableException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configures and runs the OmniEar pipeline.
 *
 * <pre>
 *   --list-devices                 list capture devices
 *   --file &lt;wav&gt;                   replay a WAV through the live pipeline
 *   --mic [device name substring]  capture from a microphone
 *   --eval &lt;dataset dir&gt;           run the evaluation harness (needs labels.csv)
 *   --eval-out &lt;path&gt;              where to write the JSON report
 *   --models &lt;dir&gt;                 model directory (default: models)
 * </pre>
 */
public final class OmniEarApplication {

    private static final Logger LOG = LoggerFactory.getLogger(OmniEarApplication.class);
    private static final int QUEUE_CAPACITY = 4;

    private final ModelConfig modelConfig;
    private final DetectionConfig detectionConfig;
    private final Path modelsDir;
    private final AlertSink sink = new ConsoleAlertSink();

    private final AtomicLong windowsProcessed = new AtomicLong();
    private final AtomicLong windowsDropped = new AtomicLong();
    private final AtomicLong alertsEmitted = new AtomicLong();

    public OmniEarApplication(ModelConfig modelConfig, DetectionConfig detectionConfig, Path modelsDir) {
        this.modelConfig = modelConfig;
        this.detectionConfig = detectionConfig;
        this.modelsDir = modelsDir;
    }

    public static void main(String[] args) throws Exception {
        String modelsDir = "models";
        String file = null;
        String device = null;
        String evalDir = null;
        String evalOut = null;
        boolean useMic = false;
        boolean listDevices = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--models" -> modelsDir = args[++i];
                case "--file" -> file = args[++i];
                case "--eval" -> evalDir = args[++i];
                case "--eval-out" -> evalOut = args[++i];
                case "--mic" -> {
                    useMic = true;
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        device = args[++i];
                    }
                }
                case "--list-devices" -> listDevices = true;
                default -> {
                    System.err.println("unknown argument: " + args[i]);
                    return;
                }
            }
        }

        if (listDevices) {
            System.out.println("Capture devices:");
            for (String name : MicrophoneSource.captureDeviceNames()) {
                System.out.println("  - " + name);
            }
            return;
        }

        ModelConfig modelConfig = ModelConfig.loadDefault();
        DetectionConfig detectionConfig = DetectionConfig.loadDefault();
        OmniEarApplication app = new OmniEarApplication(modelConfig, detectionConfig, Path.of(modelsDir));

        if (evalDir != null) {
            app.runEvaluation(Path.of(evalDir), evalOut == null ? null : Path.of(evalOut));
        } else if (file != null) {
            app.runFile(Path.of(file));
        } else if (useMic) {
            app.runMicrophone(device);
        } else {
            System.err.println("nothing to do: pass --file <wav>, --mic [device], or --eval <dir>");
        }
    }

    public void runEvaluation(Path datasetDir, Path reportOut) throws Exception {
        LOG.info("evaluating dataset {}", datasetDir.toAbsolutePath());
        EvaluationHarness harness = new EvaluationHarness(modelConfig, detectionConfig, modelsDir);
        EvaluationReport report = harness.run(datasetDir);
        report.print();
        Path out = reportOut != null ? reportOut : datasetDir.resolve("eval-report.json");
        Files.writeString(out, report.toJson());
        System.out.println("\nwrote " + out.toAbsolutePath());
    }

    public void runFile(Path wav) throws Exception {
        LOG.info("replaying {}", wav);
        try (AudioSource source = new FileAudioSource(wav)) {
            run(source, true);
        }
        printSummary("file");
    }

    public void runMicrophone(String device) throws Exception {
        AudioFormat desired = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED, modelConfig.sr, 16, 1, 2, modelConfig.sr, false);
        MicrophoneSource source;
        try {
            source = new MicrophoneSource(desired, device);
        } catch (LineUnavailableException e) {
            LOG.error("could not open microphone ({} Hz mono): {}", modelConfig.sr, e.getMessage());
            LOG.error("run --list-devices and pass a device name, or check OS microphone permissions");
            return;
        }
        Runtime.getRuntime().addShutdownHook(new Thread(source::close, "shutdown"));
        LOG.info("listening... press Ctrl+C to stop");
        run(source, false);
        printSummary("microphone");
    }

    private void run(AudioSource source, boolean waitForCompletion) throws Exception {
        BlockingQueue<AudioWindow> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
        AtomicBoolean captureEnded = new AtomicBoolean(false);

        try (CedMiniThreatModel model = CedMiniThreatModel.fromModelsDir(modelsDir, modelConfig)) {
            ScoreMapper mapper = new ScoreMapper(detectionConfig);
            TemporalDecisionEngine engine = new TemporalDecisionEngine(detectionConfig);

            AudioCaptureService capture = new AudioCaptureService(
                    source, modelConfig.sr, modelConfig.windowSamples(), modelConfig.hopSamples(),
                    w -> {
                        if (!queue.offer(w)) {
                            windowsDropped.incrementAndGet();
                        }
                    },
                    () -> captureEnded.set(true));

            Thread worker = new Thread(
                    () -> consume(queue, captureEnded, model, mapper, engine, waitForCompletion),
                    "inference");
            worker.start();
            capture.start();

            worker.join();

            if (!waitForCompletion) {
                capture.stop();
            }
            capture.close();
        }
    }

    private void consume(BlockingQueue<AudioWindow> queue,
                         AtomicBoolean captureEnded,
                         CedMiniThreatModel model,
                         ScoreMapper mapper,
                         TemporalDecisionEngine engine,
                         boolean stopWhenDrained) {
        while (true) {
            AudioWindow window;
            try {
                window = queue.poll(200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            if (window == null) {
                if (stopWhenDrained && captureEnded.get() && queue.isEmpty()) {
                    return;
                }
                if (captureEnded.get() && queue.isEmpty()) {
                    return;
                }
                continue;
            }
            process(window, model, mapper, engine);
        }
    }

    private void process(AudioWindow window,
                         CedMiniThreatModel model,
                         ScoreMapper mapper,
                         TemporalDecisionEngine engine) {
        long t0 = System.nanoTime();
        ModelPrediction prediction = model.predict(window.samples());
        long inferMs = (System.nanoTime() - t0) / 1_000_000;
        windowsProcessed.incrementAndGet();

        MappedScores mapped = mapper.map(prediction);
        List<ThreatEvent> events = engine.update(window.timestampMs(), mapped);
        for (ThreatEvent event : events) {
            long latencyMs = event.emittedWallMs() - window.capturedWallMs();
            LOG.info("event {} score={} (infer={}ms, capture->alert={}ms)",
                    event.category(), String.format("%.3f", event.score()), inferMs, latencyMs);
            sink.emit(event);
            alertsEmitted.incrementAndGet();
        }
    }

    private void printSummary(String mode) {
        System.out.printf(
                "%n[summary] mode=%s windows=%d dropped=%d alerts=%d%n",
                mode, windowsProcessed.get(), windowsDropped.get(), alertsEmitted.get());
    }
}
