package com.example.omniear.ui;

import com.example.omniear.audio.AudioCaptureService;
import com.example.omniear.audio.AudioWindow;
import com.example.omniear.audio.MicrophoneSource;
import com.example.omniear.detection.CategoryRule;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.GlassBreakDetector;
import com.example.omniear.detection.MappedScores;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
import com.example.omniear.model.AudioPreprocessor;
import com.example.omniear.model.CedMiniThreatModel;
import com.example.omniear.model.ModelConfig;
import com.example.omniear.model.ModelPrediction;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.LineUnavailableException;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OmniEar dashboard: a tabbed window with live monitoring, offline WAV testing
 * and a session alert history.
 *
 * <ul>
 *   <li><b>Dashboard</b> - device picker, mode/readiness banner, input level
 *       meter, a five-second microphone input diagnostic and per-category score
 *       bars with threshold markers.</li>
 *   <li><b>WAV Testing</b> - classify the loudest window of any WAV, or replay
 *       the whole file through the live decision pipeline.</li>
 *   <li><b>Alert History</b> - rolling alert log (P0/P1/P4 priorities) with a
 *       UTF-8 CSV export.</li>
 * </ul>
 *
 * <p>The model, decision engine and capture run on background threads; every
 * update is marshalled onto the Swing event thread.
 */
public final class OmniEarFrame extends JFrame {

    private static final Logger LOG = LoggerFactory.getLogger(OmniEarFrame.class);

    /** Windows whose strongest category score reaches this are logged for diagnosis. */
    private static final float DIAG_FLOOR = 0.2f;

    private static final Color BG = new Color(0x16161c);
    private static final Color PANEL = new Color(0x1f1f28);
    private static final Color PANEL_2 = new Color(0x272733);
    private static final Color TEXT = new Color(0xe6e6ee);
    private static final Color MUTED = new Color(0x9a9aad);
    private static final Color OK = new Color(0x2fbf71);
    private static final Color ALERT = new Color(0xe5484d);
    private static final Color WARN = new Color(0xffb74d);
    private static final DateTimeFormatter TIME = DateTimeFormatter.ofPattern("HH:mm:ss");

    private static final Map<ThreatCategory, Color> CATEGORY_COLORS = new EnumMap<>(ThreatCategory.class);
    private static final Map<ThreatCategory, String> CATEGORY_NAMES = new EnumMap<>(ThreatCategory.class);

    static {
        CATEGORY_COLORS.put(ThreatCategory.SCREAM_DISTRESS, new Color(0xff9f43));
        CATEGORY_COLORS.put(ThreatCategory.GUNSHOT, new Color(0xef5350));
        CATEGORY_COLORS.put(ThreatCategory.EXPLOSION, new Color(0xff7043));
        CATEGORY_COLORS.put(ThreatCategory.IMPACT_CRASH, new Color(0xffd54f));
        CATEGORY_COLORS.put(ThreatCategory.SIREN, new Color(0x42a5f5));
        CATEGORY_COLORS.put(ThreatCategory.GLASS_BREAK, new Color(0x26c6da));
        CATEGORY_NAMES.put(ThreatCategory.SCREAM_DISTRESS, "SCREAM / DISTRESS");
        CATEGORY_NAMES.put(ThreatCategory.GUNSHOT, "GUNSHOT");
        CATEGORY_NAMES.put(ThreatCategory.EXPLOSION, "EXPLOSION");
        CATEGORY_NAMES.put(ThreatCategory.IMPACT_CRASH, "IMPACT / CRASH");
        CATEGORY_NAMES.put(ThreatCategory.SIREN, "SIREN");
        CATEGORY_NAMES.put(ThreatCategory.GLASS_BREAK, "GLASS BREAK");
    }

    private static final GlassBreakDetector GLASS_DETECTOR = new GlassBreakDetector();

    private final ModelConfig modelConfig;
    private final DetectionConfig detectionConfig;
    private final Path modelsDir;
    private final Map<ThreatCategory, Float> thresholds = new EnumMap<>(ThreatCategory.class);

    private final JComboBox<String> deviceCombo = new JComboBox<>();
    private final JButton startStop = new JButton("Start listening");
    private final JLabel banner = new JLabel("IDLE", SwingConstants.CENTER);
    private final JLabel modeWarning = new JLabel(" ");
    private final JLabel status = new JLabel("Ready");
    private final JLabel stats = new JLabel(" ");
    private final LevelMeter levelMeter = new LevelMeter();
    private final Map<ThreatCategory, ScoreBar> bars = new EnumMap<>(ThreatCategory.class);

    private final JButton diagButton = new JButton("Run 5 s input diagnostic");
    private final JLabel diagResult = new JLabel(" ");

    private final JTextField wavPath = new JTextField();
    private final JButton wavBrowse = new JButton("Browse...");
    private final JButton wavClassify = new JButton("Classify WAV");
    private final JButton wavAnalyze = new JButton("Analyze WAV");
    private final JTextArea wavResult = new JTextArea();

    private final DefaultTableModel alertModel = new DefaultTableModel(
            new Object[] {"Time", "P", "Category", "Score", "Dur", "Label"}, 0) {
        @Override
        public boolean isCellEditable(int row, int column) {
            return false;
        }
    };
    private final JTable alertTable = new JTable(alertModel);
    private final JLabel alertCount = new JLabel("0 alerts");

    private volatile CedMiniThreatModel model;
    private volatile CedMiniThreatModel testModel;
    private volatile MicrophoneSource source;
    private volatile AudioCaptureService capture;
    private volatile Thread worker;
    private volatile boolean running;
    private final BlockingQueue<AudioWindow> queue = new ArrayBlockingQueue<>(4);
    private final AtomicLong windows = new AtomicLong();
    private final AtomicLong dropped = new AtomicLong();
    private final AtomicLong alerts = new AtomicLong();
    private final Timer bannerReset;

    public OmniEarFrame(ModelConfig modelConfig, DetectionConfig detectionConfig, Path modelsDir) {
        super("OmniEar - acoustic threat detection");
        this.modelConfig = modelConfig;
        this.detectionConfig = detectionConfig;
        this.modelsDir = modelsDir;
        for (CategoryRule rule : detectionConfig.categories) {
            thresholds.put(rule.category(), rule.threshold());
        }
        this.bannerReset = new Timer(1600, e -> showMonitoringBanner());
        bannerReset.setRepeats(false);
        buildUi();
    }

    // ---- layout -------------------------------------------------------------

    private void buildUi() {
        JPanel root = new JPanel(new BorderLayout(12, 12));
        root.setBackground(BG);
        root.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));
        root.add(buildToolbar(), BorderLayout.NORTH);

        JTabbedPane tabs = new JTabbedPane();
        tabs.setBackground(PANEL);
        tabs.setForeground(TEXT);
        tabs.setBorder(BorderFactory.createEmptyBorder(6, 0, 0, 0));
        tabs.addTab("Dashboard", buildDashboard());
        tabs.addTab("WAV Testing", buildWavTesting());
        tabs.addTab("Alert History", buildAlertHistory());
        root.add(tabs, BorderLayout.CENTER);

        root.add(buildStatusBar(), BorderLayout.SOUTH);
        setContentPane(root);
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                shutdown();
            }
        });
        setSize(820, 780);
        setMinimumSize(new Dimension(640, 620));
        setLocationRelativeTo(null);
        showIdleBanner();
    }

    private JComponent buildToolbar() {
        JPanel bar = new JPanel(new BorderLayout(10, 0));
        bar.setBackground(PANEL);
        bar.setBorder(BorderFactory.createEmptyBorder(10, 12, 10, 12));

        JLabel deviceLabel = new JLabel("Microphone");
        deviceLabel.setForeground(MUTED);
        deviceLabel.setFont(deviceLabel.getFont().deriveFont(Font.PLAIN, 12f));

        DefaultComboBoxModel<String> deviceModel = new DefaultComboBoxModel<>();
        deviceModel.addElement("<System default>");
        for (String name : MicrophoneSource.captureDeviceNames()) {
            deviceModel.addElement(name);
        }
        deviceCombo.setModel(deviceModel);
        deviceCombo.setPreferredSize(new Dimension(340, 30));
        deviceCombo.setBackground(PANEL_2);
        deviceCombo.setForeground(TEXT);

        JPanel left = new JPanel();
        left.setOpaque(false);
        left.setLayout(new BoxLayout(left, BoxLayout.X_AXIS));
        left.add(deviceLabel);
        left.add(Box.createHorizontalStrut(8));
        left.add(deviceCombo);

        startStop.setFocusPainted(false);
        startStop.setFont(startStop.getFont().deriveFont(Font.BOLD, 13f));
        startStop.setBackground(OK);
        startStop.setForeground(Color.WHITE);
        startStop.addActionListener(e -> toggle());

        bar.add(left, BorderLayout.CENTER);
        bar.add(startStop, BorderLayout.EAST);
        return bar;
    }

    private JComponent buildDashboard() {
        JPanel wrap = new JPanel();
        wrap.setBackground(BG);
        wrap.setLayout(new BoxLayout(wrap, BoxLayout.Y_AXIS));
        wrap.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));

        banner.setOpaque(true);
        banner.setFont(banner.getFont().deriveFont(Font.BOLD, 22f));
        banner.setBorder(BorderFactory.createEmptyBorder(16, 12, 16, 12));
        banner.setAlignmentX(Component.LEFT_ALIGNMENT);
        banner.setMaximumSize(new Dimension(Integer.MAX_VALUE, 64));

        modeWarning.setForeground(WARN);
        modeWarning.setFont(modeWarning.getFont().deriveFont(Font.PLAIN, 12f));
        modeWarning.setAlignmentX(Component.LEFT_ALIGNMENT);

        levelMeter.setAlignmentX(Component.LEFT_ALIGNMENT);
        levelMeter.setMaximumSize(new Dimension(Integer.MAX_VALUE, 26));

        diagResult.setForeground(MUTED);
        diagResult.setFont(diagResult.getFont().deriveFont(Font.PLAIN, 12f));
        diagResult.setAlignmentX(Component.LEFT_ALIGNMENT);

        diagButton.setFocusPainted(false);
        diagButton.setAlignmentX(Component.LEFT_ALIGNMENT);
        diagButton.addActionListener(e -> runMicDiagnostic());

        wrap.add(banner);
        wrap.add(Box.createVerticalStrut(6));
        wrap.add(modeWarning);
        wrap.add(Box.createVerticalStrut(12));

        wrap.add(sectionTitle("INPUT LEVEL"));
        wrap.add(Box.createVerticalStrut(6));
        wrap.add(levelMeter);
        wrap.add(Box.createVerticalStrut(8));
        wrap.add(diagButton);
        wrap.add(Box.createVerticalStrut(4));
        wrap.add(diagResult);
        wrap.add(Box.createVerticalStrut(16));

        wrap.add(sectionTitle("CATEGORY SCORES"));
        wrap.add(Box.createVerticalStrut(6));
        for (CategoryRule rule : detectionConfig.categories) {
            ThreatCategory cat = rule.category();
            ScoreBar bar = new ScoreBar(CATEGORY_NAMES.get(cat), CATEGORY_COLORS.get(cat));
            bar.setAlignmentX(Component.LEFT_ALIGNMENT);
            bar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
            bar.setThreshold(rule.threshold());
            bars.put(cat, bar);
            wrap.add(bar);
            wrap.add(Box.createVerticalStrut(6));
        }
        wrap.add(Box.createVerticalGlue());

        JScrollPane scroll = new JScrollPane(wrap);
        scroll.setBorder(null);
        scroll.getViewport().setBackground(BG);
        scroll.getVerticalScrollBar().setUnitIncrement(16);
        return scroll;
    }

    private JComponent buildWavTesting() {
        JPanel panel = new JPanel(new BorderLayout(0, 10));
        panel.setBackground(BG);
        panel.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));

        wavPath.setBackground(PANEL_2);
        wavPath.setForeground(TEXT);
        wavPath.setCaretColor(TEXT);

        wavBrowse.setFocusPainted(false);
        wavBrowse.addActionListener(e -> chooseWav());
        wavClassify.setFocusPainted(false);
        wavClassify.setFont(wavClassify.getFont().deriveFont(Font.BOLD, 13f));
        wavClassify.setBackground(OK);
        wavClassify.setForeground(Color.WHITE);
        wavClassify.addActionListener(e -> runWav(false));
        wavAnalyze.setFocusPainted(false);
        wavAnalyze.addActionListener(e -> runWav(true));

        JPanel picker = new JPanel(new BorderLayout(8, 0));
        picker.setBackground(BG);
        picker.add(wavPath, BorderLayout.CENTER);
        JPanel buttons = new JPanel();
        buttons.setBackground(BG);
        buttons.setLayout(new BoxLayout(buttons, BoxLayout.X_AXIS));
        buttons.add(wavBrowse);
        buttons.add(Box.createHorizontalStrut(6));
        buttons.add(wavClassify);
        buttons.add(Box.createHorizontalStrut(6));
        buttons.add(wavAnalyze);
        picker.add(buttons, BorderLayout.EAST);

        wavResult.setEditable(false);
        wavResult.setBackground(PANEL);
        wavResult.setForeground(TEXT);
        wavResult.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        wavResult.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        wavResult.setText("Pick a WAV and choose Classify (loudest 2 s window) or Analyze (whole file).\n"
                + "Classification and analysis run on a background thread.");
        JScrollPane scroll = new JScrollPane(wavResult);
        scroll.getViewport().setBackground(PANEL);
        scroll.setBorder(BorderFactory.createLineBorder(PANEL_2));

        JLabel hint = new JLabel("<html><span style='color:#9a9aad'>Classify = one loudest window through the model. "
                + "Analyze = every window through the full decision pipeline (with confirmation windows) "
                + "and lists emitted events.</span></html>");
        hint.setBorder(BorderFactory.createEmptyBorder(4, 0, 0, 0));

        panel.add(picker, BorderLayout.NORTH);
        panel.add(scroll, BorderLayout.CENTER);
        panel.add(hint, BorderLayout.SOUTH);
        return panel;
    }

    private JComponent buildAlertHistory() {
        JPanel panel = new JPanel(new BorderLayout(0, 8));
        panel.setBackground(BG);
        panel.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));

        alertTable.setBackground(PANEL);
        alertTable.setForeground(TEXT);
        alertTable.setGridColor(PANEL_2);
        alertTable.setRowHeight(26);
        alertTable.setFont(alertTable.getFont().deriveFont(Font.PLAIN, 13f));
        alertTable.getTableHeader().setReorderingAllowed(false);
        alertTable.setFillsViewportHeight(true);
        alertTable.getColumnModel().getColumn(1).setCellRenderer(new PriorityCellRenderer());
        alertTable.getColumnModel().getColumn(2).setCellRenderer(new CategoryCellRenderer());
        alertTable.getColumnModel().getColumn(1).setMaxWidth(40);

        JScrollPane scroll = new JScrollPane(alertTable);
        scroll.getViewport().setBackground(PANEL);
        scroll.setBorder(BorderFactory.createLineBorder(PANEL_2));

        JButton export = new JButton("Export CSV...");
        export.setFocusPainted(false);
        export.addActionListener(e -> exportAlerts());
        JButton clear = new JButton("Clear");
        clear.setFocusPainted(false);
        clear.addActionListener(e -> clearAlerts());

        alertCount.setForeground(MUTED);
        alertCount.setFont(alertCount.getFont().deriveFont(Font.PLAIN, 12f));

        JPanel south = new JPanel(new BorderLayout(8, 0));
        south.setBackground(BG);
        JPanel actions = new JPanel();
        actions.setBackground(BG);
        actions.setLayout(new BoxLayout(actions, BoxLayout.X_AXIS));
        actions.add(export);
        actions.add(Box.createHorizontalStrut(6));
        actions.add(clear);
        south.add(alertCount, BorderLayout.WEST);
        south.add(actions, BorderLayout.EAST);

        panel.add(scroll, BorderLayout.CENTER);
        panel.add(south, BorderLayout.SOUTH);
        return panel;
    }

    private JComponent buildStatusBar() {
        JPanel bar = new JPanel(new BorderLayout());
        bar.setOpaque(false);
        status.setForeground(MUTED);
        stats.setForeground(MUTED);
        stats.setHorizontalAlignment(SwingConstants.RIGHT);
        bar.add(status, BorderLayout.WEST);
        bar.add(stats, BorderLayout.EAST);
        return bar;
    }

    private static JLabel sectionTitle(String text) {
        JLabel label = new JLabel(text);
        label.setForeground(MUTED);
        label.setFont(label.getFont().deriveFont(Font.BOLD, 11f));
        label.setAlignmentX(Component.LEFT_ALIGNMENT);
        return label;
    }

    // ---- journal / priorities ----------------------------------------------

    /** Local alert priority: P0 critical, P1 high, P4 informational. */
    private static String priority(ThreatCategory category) {
        return switch (category) {
            case GUNSHOT, EXPLOSION -> "P0";
            case SCREAM_DISTRESS, IMPACT_CRASH, GLASS_BREAK -> "P1";
            case SIREN -> "P4";
            default -> "--";
        };
    }

    private static Color priorityColor(String p) {
        return switch (p) {
            case "P0" -> ALERT;
            case "P1" -> WARN;
            case "P4" -> new Color(0x42a5f5);
            default -> MUTED;
        };
    }

    // ---- pipeline -----------------------------------------------------------

    /** Selects the first device whose name contains {@code substring} (case-insensitive). */
    public void selectDevice(String substring) {
        if (substring == null) {
            return;
        }
        String needle = substring.toLowerCase();
        for (int i = 0; i < deviceCombo.getItemCount(); i++) {
            String item = deviceCombo.getItemAt(i);
            if (item != null && item.toLowerCase().contains(needle)) {
                deviceCombo.setSelectedIndex(i);
                return;
            }
        }
    }

    /** Starts listening if currently idle. Intended to be called on the EDT. */
    public void startListening() {
        if (!running && startStop.isEnabled()) {
            startPipeline();
        }
    }

    private void toggle() {
        if (running) {
            stopPipeline("Stopped");
        } else {
            startPipeline();
        }
    }

    private void startPipeline() {
        String deviceName = selectedDevice();
        startStop.setEnabled(false);
        status.setText("Loading model...");
        modeWarning.setText("Loading model...");
        Thread starter = new Thread(() -> {
            try {
                CedMiniThreatModel loaded = CedMiniThreatModel.fromModelsDir(modelsDir, modelConfig);
                AudioFormat desired = new AudioFormat(
                        AudioFormat.Encoding.PCM_SIGNED, modelConfig.sr, 16, 1, 2, modelConfig.sr, false);
                MicrophoneSource mic = new MicrophoneSource(desired, deviceName);
                model = loaded;
                source = mic;
                SwingUtilities.invokeLater(this::beginCapture);
            } catch (LineUnavailableException e) {
                SwingUtilities.invokeLater(() -> fail("Could not open microphone: " + e.getMessage()
                        + "  (try another device or check OS mic permission)"));
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> fail("Could not start: " + e.getMessage()));
            }
        }, "gui-start");
        starter.setDaemon(true);
        starter.start();
    }

    private void beginCapture() {
        ScoreMapper mapper = new ScoreMapper(detectionConfig);
        TemporalDecisionEngine engine = new TemporalDecisionEngine(detectionConfig);
        running = true;
        queue.clear();
        windows.set(0);
        dropped.set(0);
        alerts.set(0);

        capture = new AudioCaptureService(source, modelConfig.sr,
                modelConfig.windowSamples(), modelConfig.hopSamples(),
                w -> {
                    if (!queue.offer(w)) {
                        dropped.incrementAndGet();
                    }
                },
                () -> SwingUtilities.invokeLater(this::onCaptureEnded));

        worker = new Thread(() -> consume(mapper, engine), "gui-inference");
        worker.setDaemon(true);
        capture.start();
        worker.start();

        setRunningUi(true);
        status.setText("Listening - make some noise");
    }

    private void consume(ScoreMapper mapper, TemporalDecisionEngine engine) {
        while (running) {
            AudioWindow window;
            try {
                window = queue.poll(200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            if (window == null) {
                continue;
            }
            float[] raw = window.samples();
            ModelPrediction prediction = model.predict(AudioPreprocessor.normalizePeak(raw));
            MappedScores mapped = mapper.map(prediction);
            if (GLASS_DETECTOR.detects(raw)) {
                mapped = mapped.withScore(ThreatCategory.GLASS_BREAK, 1f, "glass(hf)");
            }
            List<ThreatEvent> events = engine.update(window.timestampMs(), mapped);
            windows.incrementAndGet();

            float db = (float) (20 * Math.log10(Math.max(rms(raw), 1e-6)));
            float peakDb = (float) (20 * Math.log10(Math.max(peak(raw), 1e-6)));
            float level = normalizeDb(db);
            float peakLevel = normalizeDb(peakDb);
            if (mapped.strongestScore() >= DIAG_FLOOR) {
                logDiagnostics(prediction, mapped, db, peakDb);
            }
            MappedScores mappedForUi = mapped;
            SwingUtilities.invokeLater(() -> updateLive(level, peakLevel, mappedForUi, events));
        }
    }

    private void updateLive(float level, float peakLevel, MappedScores mapped, List<ThreatEvent> events) {
        levelMeter.setLevels(level, peakLevel);
        for (Map.Entry<ThreatCategory, ScoreBar> e : bars.entrySet()) {
            e.getValue().setScore(mapped.score(e.getKey()));
        }
        for (ThreatEvent event : events) {
            alerts.incrementAndGet();
            addAlertRow(event);
            flashAlert(event);
        }
        stats.setText(String.format("windows=%d  dropped=%d  alerts=%d",
                windows.get(), dropped.get(), alerts.get()));
    }

    private void logDiagnostics(ModelPrediction prediction, MappedScores mapped, float rmsDb, float peakDb) {
        StringBuilder cats = new StringBuilder();
        for (CategoryRule rule : detectionConfig.categories) {
            cats.append(rule.category().name()).append('=')
                    .append(String.format("%.2f", mapped.score(rule.category()))).append(' ');
        }
        StringBuilder top = new StringBuilder();
        for (Map.Entry<String, Float> e : prediction.topK(3)) {
            top.append(e.getKey()).append('=').append(String.format("%.3f", e.getValue())).append(' ');
        }
        LOG.info("diag rms={}dB peak={}dB strongest={} [{}] top: {}",
                String.format("%.0f", rmsDb),
                String.format("%.0f", peakDb),
                String.format("%.3f", mapped.strongestScore()),
                cats.toString().trim(),
                top.toString().trim());
    }

    private void addAlertRow(ThreatEvent event) {
        String time = Instant.ofEpochMilli(event.emittedWallMs())
                .atZone(ZoneId.systemDefault()).toLocalTime().format(TIME);
        alertModel.insertRow(0, new Object[] {
                time,
                priority(event.category()),
                CATEGORY_NAMES.getOrDefault(event.category(), event.category().name()),
                String.format("%.2f", event.score()),
                event.durationMs() + " ms",
                event.drivingLabel()});
        while (alertModel.getRowCount() > 1000) {
            alertModel.removeRow(alertModel.getRowCount() - 1);
        }
        alertCount.setText(alertModel.getRowCount() + " alerts");
    }

    private void flashAlert(ThreatEvent event) {
        String name = CATEGORY_NAMES.getOrDefault(event.category(), event.category().name());
        banner.setText(String.format("%s  \u26a0  %s   %.2f",
                priority(event.category()), name, event.score()));
        banner.setBackground(ALERT);
        banner.setForeground(Color.WHITE);
        bannerReset.restart();
    }

    private void showMonitoringBanner() {
        if (running) {
            banner.setText("\u25cf  MONITORING  " + prioritySummary());
            banner.setBackground(new Color(0x15351f));
            banner.setForeground(new Color(0x7ee2a8));
        } else {
            showIdleBanner();
        }
    }

    private static String prioritySummary() {
        return "P0 gunshot/explosion \u00b7 P1 scream/impact/glass \u00b7 P4 siren";
    }

    private void showIdleBanner() {
        banner.setText("IDLE");
        banner.setBackground(PANEL);
        banner.setForeground(MUTED);
    }

    private void stopPipeline(String message) {
        running = false;
        AudioCaptureService cap = capture;
        Thread w = worker;
        if (cap != null) {
            cap.close();
        }
        if (w != null) {
            try {
                w.join(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        closeModel();
        capture = null;
        source = null;
        worker = null;
        setRunningUi(false);
        status.setText(message);
        stats.setText(String.format("windows=%d  dropped=%d  alerts=%d",
                windows.get(), dropped.get(), alerts.get()));
    }

    private void closeModel() {
        CedMiniThreatModel m = model;
        if (m != null) {
            try {
                m.close();
            } catch (RuntimeException ignored) {
                // best effort on shutdown
            }
        }
        model = null;
    }

    private void closeTestModel() {
        CedMiniThreatModel m = testModel;
        if (m != null) {
            try {
                m.close();
            } catch (RuntimeException ignored) {
                // best effort on shutdown
            }
        }
        testModel = null;
    }

    private void onCaptureEnded() {
        if (running) {
            stopPipeline("Capture ended");
        }
    }

    private void fail(String message) {
        startStop.setEnabled(true);
        status.setText(message);
        modeWarning.setText("Not monitoring");
        showIdleBanner();
        JOptionPane.showMessageDialog(this, message, "OmniEar", JOptionPane.ERROR_MESSAGE);
    }

    private void setRunningUi(boolean isRunning) {
        startStop.setEnabled(true);
        startStop.setText(isRunning ? "Stop" : "Start listening");
        startStop.setBackground(isRunning ? ALERT : OK);
        deviceCombo.setEnabled(!isRunning);
        diagButton.setEnabled(!isRunning);
        modeWarning.setText(isRunning
                ? "Live monitoring from " + deviceNameForWarning() + ". Audio is processed in memory and never saved."
                : "Idle - press Start listening to monitor. The model is not loaded until you start.");
        if (isRunning) {
            showMonitoringBanner();
        }
    }

    private String deviceNameForWarning() {
        Object item = deviceCombo.getSelectedItem();
        if (item == null || "<System default>".equals(item)) {
            return "the system default input";
        }
        return item.toString();
    }

    private void shutdown() {
        running = false;
        AudioCaptureService cap = capture;
        if (cap != null) {
            cap.close();
        }
        closeModel();
        closeTestModel();
        dispose();
        System.exit(0);
    }

    private String selectedDevice() {
        Object item = deviceCombo.getSelectedItem();
        if (item == null || "<System default>".equals(item)) {
            return null;
        }
        return item.toString();
    }

    private static float normalizeDb(float db) {
        return Math.max(0f, Math.min(1f, (db + 60f) / 60f));
    }

    private static float rms(float[] samples) {
        double sum = 0;
        for (float v : samples) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum / Math.max(1, samples.length));
    }

    private static float peak(float[] samples) {
        float max = 0f;
        for (float v : samples) {
            max = Math.max(max, Math.abs(v));
        }
        return max;
    }

    // ---- microphone diagnostic ---------------------------------------------

    private void runMicDiagnostic() {
        if (running) {
            JOptionPane.showMessageDialog(this, "Stop listening before running the input diagnostic.",
                    "OmniEar", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        String device = selectedDevice();
        diagButton.setEnabled(false);
        diagResult.setText("Recording 5 s...");
        Thread t = new Thread(() -> {
            String result;
            try {
                result = measureInput(device, 5.0);
            } catch (LineUnavailableException e) {
                result = "Could not open microphone: " + e.getMessage()
                        + " (try another device or check OS permission)";
            } catch (Exception e) {
                result = "Diagnostic failed: " + e.getMessage();
            }
            String r = result;
            SwingUtilities.invokeLater(() -> {
                diagResult.setText(r);
                diagButton.setEnabled(!running);
            });
        }, "mic-diag");
        t.setDaemon(true);
        t.start();
    }

    /** Reads {@code seconds} of mono input and reports level/clipping statistics. */
    private String measureInput(String device, double seconds) throws Exception {
        AudioFormat desired = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED, modelConfig.sr, 16, 1, 2, modelConfig.sr, false);
        try (MicrophoneSource mic = new MicrophoneSource(desired, device)) {
            AudioFormat fmt = mic.format();
            int channels = Math.max(1, fmt.getChannels());
            int frameSize = 2 * channels;
            byte[] buf = new byte[4096];
            double sumSq = 0;
            double peak = 0;
            long count = 0;
            long nearFull = 0;
            long deadline = System.nanoTime() + (long) (seconds * 1_000_000_000L);
            while (System.nanoTime() < deadline) {
                int n = mic.read(buf, 0, buf.length);
                if (n <= 0) {
                    continue;
                }
                for (int f = 0; f + frameSize <= n; f += frameSize) {
                    float sum = 0f;
                    for (int c = 0; c < channels; c++) {
                        int lo = buf[f + c * 2] & 0xFF;
                        int hi = buf[f + c * 2 + 1]; // signed high byte
                        sum += ((hi << 8) | lo) / 32768f;
                    }
                    float v = sum / channels;
                    sumSq += (double) v * v;
                    float a = Math.abs(v);
                    if (a > peak) {
                        peak = a;
                    }
                    if (a >= 0.98f) {
                        nearFull++;
                    }
                    count++;
                }
            }
            if (count == 0) {
                return "No audio captured.";
            }
            double rms = Math.sqrt(sumSq / count);
            double rmsDb = 20 * Math.log10(Math.max(rms, 1e-9));
            double peakDb = 20 * Math.log10(Math.max(peak, 1e-9));
            double nearPct = 100.0 * nearFull / count;
            String guidance;
            if (peakDb > -1.0 || nearPct > 1.0) {
                guidance = "too hot - lower the OS microphone gain";
            } else if (peakDb < -30.0) {
                guidance = "very quiet - raise the OS gain or move closer";
            } else if (peakDb < -20.0) {
                guidance = "a little low - consider raising the OS gain";
            } else {
                guidance = "level looks good";
            }
            return String.format(
                    "RMS %.1f dBFS   peak %.1f dBFS   near-full-scale %.2f%%   -> %s",
                    rmsDb, peakDb, nearPct, guidance);
        }
    }

    // ---- WAV testing --------------------------------------------------------

    private void chooseWav() {
        JFileChooser chooser = new JFileChooser(wavPath.getText().isBlank()
                ? new java.io.File(".")
                : new java.io.File(wavPath.getText()).getParentFile());
        chooser.setFileFilter(new FileNameExtensionFilter("WAV audio (*.wav)", "wav"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            wavPath.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    private void runWav(boolean analyzeAll) {
        String path = wavPath.getText().trim();
        if (path.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Choose a WAV file first.",
                    "OmniEar", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        wavClassify.setEnabled(false);
        wavAnalyze.setEnabled(false);
        wavResult.setText((analyzeAll ? "Analyzing " : "Classifying ") + path + " ...");
        Thread t = new Thread(() -> {
            String out;
            try {
                out = analyzeAll ? analyzeWavFile(Path.of(path)) : classifyWavFile(Path.of(path));
            } catch (Exception e) {
                LOG.warn("WAV test failed for {}", path, e);
                out = "Failed: " + e.getMessage();
            }
            String text = out;
            SwingUtilities.invokeLater(() -> {
                wavResult.setText(text);
                wavResult.setCaretPosition(0);
                wavClassify.setEnabled(true);
                wavAnalyze.setEnabled(true);
            });
        }, "wav-test");
        t.setDaemon(true);
        t.start();
    }

    /** Lazily loads a dedicated model for offline WAV testing (kept out of the live worker). */
    private CedMiniThreatModel testModel() throws Exception {
        CedMiniThreatModel m = testModel;
        if (m == null) {
            synchronized (this) {
                if (testModel == null) {
                    testModel = CedMiniThreatModel.fromModelsDir(modelsDir, modelConfig);
                }
                m = testModel;
            }
        }
        return m;
    }

    private String classifyWavFile(Path file) throws Exception {
        int sr = modelConfig.sr;
        float[] samples = AudioPreprocessor.readMono(file, sr);
        StringBuilder sb = new StringBuilder();
        sb.append("FILE     ").append(file.getFileName()).append('\n');
        sb.append(String.format("LENGTH   %.2f s @ %d Hz (%d samples)%n",
                samples.length / (double) sr, sr, samples.length));
        if (samples.length == 0) {
            sb.append("\nNo audio samples in file.");
            return sb.toString();
        }
        int win = modelConfig.windowSamples();
        int off = loudestOffset(samples, win, modelConfig.hopSamples());
        float[] window = AudioPreprocessor.window(samples, off, win);
        sb.append(String.format("WINDOW   loudest %.1f s at t=%.2f s%n",
                win / (double) sr, off / (double) sr));

        ModelPrediction prediction = testModel().predict(AudioPreprocessor.normalizePeak(window));
        MappedScores mapped = mapWithGlass(prediction, window);

        sb.append("\nMODEL TOP-5\n");
        int rank = 0;
        for (Map.Entry<String, Float> e : prediction.topK(5)) {
            sb.append(String.format("  %d. %-14s %.3f%n", ++rank, e.getKey(), e.getValue()));
        }
        sb.append("\nCATEGORY SCORES\n");
        for (CategoryRule rule : detectionConfig.categories) {
            ThreatCategory cat = rule.category();
            float score = mapped.score(cat);
            boolean hot = score >= rule.threshold();
            sb.append(String.format("  %-16s %.3f  (thr %.3f)%s%n",
                    CATEGORY_NAMES.get(cat), score, rule.threshold(), hot ? "  <-- ALERT" : ""));
        }
        ThreatCategory best = mapped.strongest();
        float bestScore = mapped.strongestScore();
        boolean alert = bestScore >= thresholds.getOrDefault(best, 1f) && best != ThreatCategory.NORMAL;
        sb.append("\nVERDICT  ").append(CATEGORY_NAMES.getOrDefault(best, best.name()))
                .append("   priority ").append(priority(best))
                .append(String.format("   score %.3f", bestScore))
                .append(alert ? "   -> ALERT" : "   -> below threshold");
        return sb.toString();
    }

    private String analyzeWavFile(Path file) throws Exception {
        int sr = modelConfig.sr;
        int win = modelConfig.windowSamples();
        int hop = modelConfig.hopSamples();
        float[] samples = AudioPreprocessor.readMono(file, sr);
        StringBuilder sb = new StringBuilder();
        sb.append("FILE     ").append(file.getFileName()).append('\n');
        sb.append(String.format("LENGTH   %.2f s @ %d Hz%n", samples.length / (double) sr, sr));
        if (samples.length == 0) {
            sb.append("\nNo audio samples in file.");
            return sb.toString();
        }
        float[][] wins = AudioPreprocessor.windows(samples, modelConfig);
        sb.append("WINDOWS  ").append(wins.length).append(String.format("  (%.1f s / %.1f s hop)%n",
                win / (double) sr, hop / (double) sr));

        CedMiniThreatModel m = testModel();
        ScoreMapper mapper = new ScoreMapper(detectionConfig);
        TemporalDecisionEngine engine = new TemporalDecisionEngine(detectionConfig);
        List<ThreatEvent> events = new ArrayList<>();
        for (int i = 0; i < wins.length; i++) {
            ModelPrediction prediction = m.predict(AudioPreprocessor.normalizePeak(wins[i]));
            MappedScores mapped = mapWithGlass(prediction, wins[i]);
            long endMs = (long) ((i * hop + win) * 1000L / sr);
            events.addAll(engine.update(endMs, mapped));
        }
        sb.append("\nEVENTS   ").append(events.size()).append('\n');
        for (ThreatEvent event : events) {
            sb.append(String.format("  %6.2f s  %-16s %s  score %.3f  via %s%n",
                    event.endMs() / 1000.0,
                    CATEGORY_NAMES.getOrDefault(event.category(), event.category().name()),
                    priority(event.category()),
                    event.score(),
                    event.drivingLabel()));
        }
        if (events.isEmpty()) {
            sb.append("  (no category crossed its confirmation threshold)");
        }
        return sb.toString();
    }

    private MappedScores mapWithGlass(ModelPrediction prediction, float[] window) {
        MappedScores mapped = new ScoreMapper(detectionConfig).map(prediction);
        if (GLASS_DETECTOR.detects(window)) {
            mapped = mapped.withScore(ThreatCategory.GLASS_BREAK, 1f, "glass(hf)");
        }
        return mapped;
    }

    /** Offset of the highest-energy window of length {@code win}, stepping by {@code hop}. */
    private static int loudestOffset(float[] samples, int win, int hop) {
        if (samples.length <= win) {
            return 0;
        }
        int best = 0;
        double bestEnergy = -1;
        for (int o = 0; o + win <= samples.length; o += Math.max(1, hop)) {
            double sum = 0;
            for (int i = 0; i < win; i++) {
                float v = samples[o + i];
                sum += (double) v * v;
            }
            if (sum > bestEnergy) {
                bestEnergy = sum;
                best = o;
            }
        }
        return best;
    }

    // ---- alert history ------------------------------------------------------

    private void clearAlerts() {
        alertModel.setRowCount(0);
        alertCount.setText("0 alerts");
    }

    private void exportAlerts() {
        if (alertModel.getRowCount() == 0) {
            JOptionPane.showMessageDialog(this, "There are no alerts to export.",
                    "OmniEar", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        JFileChooser chooser = new JFileChooser(".");
        chooser.setSelectedFile(new java.io.File("omniear-alerts.csv"));
        if (chooser.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        Path out = chooser.getSelectedFile().toPath();
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < alertModel.getColumnCount(); c++) {
            if (c > 0) {
                sb.append(',');
            }
            sb.append(csv(alertModel.getColumnName(c)));
        }
        sb.append('\n');
        for (int r = 0; r < alertModel.getRowCount(); r++) {
            for (int c = 0; c < alertModel.getColumnCount(); c++) {
                if (c > 0) {
                    sb.append(',');
                }
                Object v = alertModel.getValueAt(r, c);
                sb.append(csv(v == null ? "" : v.toString()));
            }
            sb.append('\n');
        }
        try {
            Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
            status.setText("Exported " + alertModel.getRowCount() + " alerts to " + out);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this, "Could not write " + out + ": " + e.getMessage(),
                    "OmniEar", JOptionPane.ERROR_MESSAGE);
        }
    }

    private static String csv(String value) {
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    // ---- components ---------------------------------------------------------

    private static final class LevelMeter extends JComponent {
        private volatile float level;
        private volatile float peak;

        LevelMeter() {
            setOpaque(false);
            setPreferredSize(new Dimension(100, 24));
        }

        void setLevels(float level, float peak) {
            this.level = level;
            this.peak = peak;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            int w = getWidth();
            int h = getHeight();
            g2.setColor(PANEL);
            g2.fillRoundRect(0, 0, w, h, h, h);
            int fill = (int) (level * (w - 2));
            if (fill > 0) {
                g2.setColor(levelColor(level));
                g2.fillRoundRect(1, 1, fill, h - 2, h - 2, h - 2);
            }
            int px = 1 + (int) (peak * (w - 2));
            g2.setColor(new Color(0xffffff));
            g2.fillRect(Math.min(px, w - 2), 1, 2, h - 2);
            g2.dispose();
        }

        private static Color levelColor(float level) {
            if (level > 0.85f) {
                return ALERT;
            }
            if (level > 0.6f) {
                return new Color(0xffd54f);
            }
            return OK;
        }
    }

    private static final class ScoreBar extends JComponent {
        private final String name;
        private final Color color;
        private volatile float score;
        private volatile float threshold;

        ScoreBar(String name, Color color) {
            this.name = name;
            this.color = color;
            setOpaque(false);
            setPreferredSize(new Dimension(100, 28));
        }

        void setThreshold(float threshold) {
            this.threshold = threshold;
            repaint();
        }

        void setScore(float score) {
            this.score = score;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            int w = getWidth();
            int h = getHeight();
            g2.setColor(PANEL);
            g2.fillRoundRect(0, 0, w, h, 8, 8);

            boolean hot = score >= threshold;
            int fill = (int) (Math.min(score, 1f) * (w - 2));
            if (fill > 0) {
                g2.setColor(hot ? color : color.darker().darker());
                g2.fillRoundRect(1, 1, fill, h - 2, 7, 7);
            }

            int tick = (int) (Math.min(threshold, 1f) * (w - 2));
            g2.setColor(new Color(0xf0f0f5));
            g2.fillRect(tick + 1, 3, 2, h - 6);

            g2.setFont(getFont().deriveFont(Font.BOLD, 12f));
            g2.setColor(hot ? Color.WHITE : TEXT);
            g2.drawString(name, 10, h / 2 + 5);
            String value = String.format("%.2f", score);
            int vw = g2.getFontMetrics().stringWidth(value);
            g2.drawString(value, w - vw - 10, h / 2 + 5);
            g2.dispose();
        }
    }

    private static final class CategoryCellRenderer extends DefaultTableCellRenderer {
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                       boolean hasFocus, int row, int column) {
            Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
            c.setBackground(isSelected ? table.getSelectionBackground() : PANEL);
            c.setForeground(TEXT);
            if (value != null) {
                for (Map.Entry<ThreatCategory, String> e : CATEGORY_NAMES.entrySet()) {
                    if (e.getValue().equals(value)) {
                        c.setForeground(CATEGORY_COLORS.get(e.getKey()));
                        break;
                    }
                }
            }
            return c;
        }
    }

    private static final class PriorityCellRenderer extends DefaultTableCellRenderer {
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                       boolean hasFocus, int row, int column) {
            Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
            c.setBackground(isSelected ? table.getSelectionBackground() : PANEL);
            setHorizontalAlignment(SwingConstants.CENTER);
            setFont(getFont().deriveFont(Font.BOLD, 12f));
            c.setForeground(value == null ? MUTED : priorityColor(value.toString()));
            return c;
        }
    }
}
