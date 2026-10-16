package com.example.omniear.app;

import com.example.omniear.alert.AlertSink;
import com.example.omniear.alert.ConsoleAlertSink;
import com.example.omniear.audio.AudioCaptureService;
import com.example.omniear.audio.AudioSource;
import com.example.omniear.audio.AudioWindow;
import com.example.omniear.audio.FileAudioSource;
import com.example.omniear.audio.MicrophoneSource;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.GlassBreakDetector;
import com.example.omniear.detection.MappedScores;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.eval.EvaluationHarness;
import com.example.omniear.eval.EvaluationReport;
import com.example.omniear.model.AudioPreprocessor;
import com.example.omniear.model.CedMiniThreatModel;
import com.example.omniear.model.ModelConfig;
import com.example.omniear.model.ModelPrediction;
import com.example.omniear.ui.OmniEarFrame;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configures and runs the OmniEar pipeline.
 *
 * <pre>
 *   --list-devices                 list capture devices
 *   --file &lt;wav&gt;                   replay a WAV through the live pipeline
 *   --mic [device name substring]  capture from a microphone (console)
 *   --gui [device name substring]  live dashboard window
 *   --record &lt;wav&gt;                 capture mic audio to a WAV (see --seconds)
 *   --seconds &lt;n&gt;                  record length for --record (default 6)
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

    private final GlassBreakDetector glassBreakDetector = new GlassBreakDetector();

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
        String recordPath = null;
        double seconds = 6.0;
        int captureRate = 0;
        boolean useMic = false;
        boolean useGui = false;
        boolean listDevices = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--models" -> modelsDir = args[++i];
                case "--file" -> file = args[++i];
                case "--eval" -> evalDir = args[++i];
                case "--eval-out" -> evalOut = args[++i];
                case "--record" -> {
                    recordPath = args[++i];
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        device = args[++i];
                    }
                }
                case "--seconds" -> seconds = Double.parseDouble(args[++i]);
                case "--rate" -> captureRate = Integer.parseInt(args[++i]);
                case "--mic" -> {
                    useMic = true;
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        device = args[++i];
                    }
                }
                case "--gui" -> {
                    useGui = true;
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
        } else if (recordPath != null) {
            app.runRecord(Path.of(recordPath), seconds, device, captureRate);
        } else if (useMic) {
            app.runMicrophone(device);
        } else if (useGui) {
            app.runGui(device);
        } else {
            System.err.println("nothing to do: pass --file <wav>, --mic [device], --gui [device], or --eval <dir>");
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

    public void runRecord(Path out, double seconds, String device, int captureRate) throws Exception {
        float rate = captureRate > 0 ? captureRate : modelConfig.sr;
        AudioFormat desired = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED, rate, 16, 1, 2, rate, false);
        LOG.info("recording {}s from {} ...", seconds, device == null ? "default device" : device);
        try (MicrophoneSource source = new MicrophoneSource(desired, device);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[4096];
            long deadline = System.nanoTime() + (long) (seconds * 1_000_000_000L);
            while (System.nanoTime() < deadline) {
                int n = source.read(buf, 0, buf.length);
                if (n > 0) {
                    bos.write(buf, 0, n);
                }
            }
            byte[] data = bos.toByteArray();
            AudioFormat fmt = source.format();
            int frameSize = fmt.getFrameSize();
            try (AudioInputStream ais =
                         new AudioInputStream(new ByteArrayInputStream(data), fmt, data.length / frameSize)) {
                AudioSystem.write(ais, AudioFileFormat.Type.WAVE, out.toFile());
            }
            System.out.println("wrote " + out.toAbsolutePath()
                    + "  frames=" + (data.length / frameSize)
                    + "  rate=" + fmt.getSampleRate()
                    + "  channels=" + fmt.getChannels());
        }
    }

    public void runGui(String device) throws Exception {
        try {
            UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
        } catch (Exception ignored) {
            // fall back to the default look and feel
        }
        SwingUtilities.invokeAndWait(() -> {
            OmniEarFrame frame = new OmniEarFrame(modelConfig, detectionConfig, modelsDir);
            frame.selectDevice(device);
            frame.setVisible(true);
            frame.startListening();
        });
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
        // File replay must be lossless (every window evaluated); live capture drops
        // stale windows so inference can keep up with real time.
        BlockingQueue<AudioWindow> queue = waitForCompletion
                ? new LinkedBlockingQueue<>()
                : new ArrayBlockingQueue<>(QUEUE_CAPACITY);
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
        ModelPrediction prediction = model.predict(AudioPreprocessor.normalizePeak(window.samples()));
        long inferMs = (System.nanoTime() - t0) / 1_000_000;
        windowsProcessed.incrementAndGet();

        MappedScores mapped = mapper.map(prediction);
        if (glassBreakDetector.detects(window.samples())) {
            mapped = mapped.withScore(ThreatCategory.GLASS_BREAK, 1f, "glass(hf)");
            LOG.debug("glass-break detector fired (t={}ms)", window.timestampMs());
        }
        List<ThreatEvent> events =
                engine.update(window.timestampMs(), mapped, AudioPreprocessor.rmsDb(window.samples()));
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
