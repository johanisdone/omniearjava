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
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.LayoutManager;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
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
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.Scrollable;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * OmniEar dashboard: a dark, card-based window with live monitoring and a
 * session alert history.
 *
 * <ul>
 *   <li><b>Dashboard</b> - device picker, mode/readiness banner, input level
 *       meter, a five-second microphone input diagnostic and per-category score
 *       bars with threshold markers.</li>
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

    // ---- palette ------------------------------------------------------------
    private static final Color BG = new Color(0x0e1015);
    private static final Color CARD = new Color(0x171a22);
    private static final Color CARD_2 = new Color(0x1f2430);
    private static final Color BORDER = new Color(0x2a3040);
    private static final Color TEXT = new Color(0xe8eaf0);
    private static final Color MUTED = new Color(0x8b90a3);
    private static final Color ACCENT = new Color(0x5b8cff);
    private static final Color OK = new Color(0x35d07f);
    private static final Color ALERT = new Color(0xff5c5c);
    private static final Color WARN = new Color(0xffb020);
    private static final DateTimeFormatter TIME = DateTimeFormatter.ofPattern("HH:mm:ss");

    private static final String FONT_FAMILY = pickFontFamily();

    private static final Map<ThreatCategory, Color> CATEGORY_COLORS = new EnumMap<>(ThreatCategory.class);
    private static final Map<ThreatCategory, String> CATEGORY_NAMES = new EnumMap<>(ThreatCategory.class);

    static {
        CATEGORY_COLORS.put(ThreatCategory.SCREAM_DISTRESS, new Color(0xff9f43));
        CATEGORY_COLORS.put(ThreatCategory.GUNSHOT, new Color(0xef5350));
        CATEGORY_COLORS.put(ThreatCategory.EXPLOSION, new Color(0xff7043));
        CATEGORY_COLORS.put(ThreatCategory.IMPACT_CRASH, new Color(0xffd54f));
        CATEGORY_COLORS.put(ThreatCategory.SIREN, new Color(0x5b8cff));
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
    private final RoundedButton startStop = new RoundedButton("Start listening", OK, 12);
    private final StatusBanner banner = new StatusBanner();
    private final JLabel modeWarning = new JLabel(" ");
    private final JLabel status = new JLabel("Ready");
    private final JLabel stats = new JLabel(" ");
    private final LevelMeter levelMeter = new LevelMeter();
    private final Map<ThreatCategory, ScoreBar> bars = new EnumMap<>(ThreatCategory.class);

    private final RoundedButton diagButton = new RoundedButton("Run 5 s input diagnostic", CARD_2, 10);
    private final JLabel diagResult = new JLabel(" ");

    private final CardLayout cards = new CardLayout();
    private final JPanel cardHost = new JPanel(cards);
    private final RoundedButton navDashboard = new RoundedButton("Dashboard", ACCENT, 12);
    private final RoundedButton navAlerts = new RoundedButton("Alerts", CARD_2, 12);

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

    // ---- fonts --------------------------------------------------------------

    private static String pickFontFamily() {
        java.util.Set<String> available = new java.util.HashSet<>(List.of(
                java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames()));
        for (String candidate : new String[] {"Segoe UI", "Inter", "Roboto", "Helvetica Neue", "SansSerif"}) {
            if (available.contains(candidate)) {
                return candidate;
            }
        }
        return "SansSerif";
    }

    private static Font font(int style, float size) {
        return new Font(FONT_FAMILY, style, 12).deriveFont(size);
    }

    // ---- layout -------------------------------------------------------------

    private void buildUi() {
        JPanel root = new JPanel(new BorderLayout(0, 14));
        root.setBackground(BG);
        root.setBorder(BorderFactory.createEmptyBorder(16, 16, 14, 16));
        root.add(buildHeader(), BorderLayout.NORTH);

        cardHost.setOpaque(false);
        cardHost.add(buildDashboard(), "dashboard");
        cardHost.add(buildAlertHistory(), "alerts");
        root.add(cardHost, BorderLayout.CENTER);

        root.add(buildStatusBar(), BorderLayout.SOUTH);
        setContentPane(root);
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                shutdown();
            }
        });
        setSize(900, 840);
        setMinimumSize(new Dimension(680, 640));
        setLocationRelativeTo(null);
        showCard("dashboard");
        showIdleBanner();
    }

    private JComponent buildHeader() {
        RoundedPanel card = new RoundedPanel(new BorderLayout(0, 16), CARD, 16, BORDER);
        card.setBorder(BorderFactory.createEmptyBorder(18, 20, 18, 20));

        JLabel title = new JLabel("<html><span style='color:#5b8cff'>Omni</span>"
                + "<span style='color:#e8eaf0'>Ear</span></html>");
        title.setFont(font(Font.BOLD, 24f));
        JLabel subtitle = new JLabel("Real-time acoustic threat detection");
        subtitle.setForeground(MUTED);
        subtitle.setFont(font(Font.PLAIN, 12f));

        JPanel titleBlock = transparent();
        titleBlock.setLayout(new BoxLayout(titleBlock, BoxLayout.Y_AXIS));
        title.setAlignmentX(Component.LEFT_ALIGNMENT);
        subtitle.setAlignmentX(Component.LEFT_ALIGNMENT);
        titleBlock.add(title);
        titleBlock.add(Box.createVerticalStrut(2));
        titleBlock.add(subtitle);

        navDashboard.setFont(font(Font.BOLD, 13f));
        navAlerts.setFont(font(Font.BOLD, 13f));
        navDashboard.addActionListener(e -> showCard("dashboard"));
        navAlerts.addActionListener(e -> showCard("alerts"));
        JPanel nav = transparent();
        nav.setLayout(new BoxLayout(nav, BoxLayout.X_AXIS));
        nav.add(navDashboard);
        nav.add(Box.createHorizontalStrut(8));
        nav.add(navAlerts);

        JPanel top = transparent(new BorderLayout(12, 0));
        top.add(titleBlock, BorderLayout.WEST);
        top.add(nav, BorderLayout.EAST);

        JLabel deviceLabel = new JLabel("Microphone");
        deviceLabel.setForeground(MUTED);
        deviceLabel.setFont(font(Font.PLAIN, 12f));

        DefaultComboBoxModel<String> deviceModel = new DefaultComboBoxModel<>();
        deviceModel.addElement("<System default>");
        for (String name : MicrophoneSource.captureDeviceNames()) {
            deviceModel.addElement(name);
        }
        deviceCombo.setModel(deviceModel);
        deviceCombo.setPreferredSize(new Dimension(340, 34));
        deviceCombo.setBackground(CARD_2);
        deviceCombo.setForeground(TEXT);
        deviceCombo.setFont(font(Font.PLAIN, 12f));

        JPanel left = transparent();
        left.setLayout(new BoxLayout(left, BoxLayout.X_AXIS));
        left.add(deviceLabel);
        left.add(Box.createHorizontalStrut(10));
        left.add(deviceCombo);

        startStop.setFont(font(Font.BOLD, 13f));
        startStop.addActionListener(e -> toggle());

        JPanel controls = transparent(new BorderLayout());
        controls.add(left, BorderLayout.WEST);
        controls.add(startStop, BorderLayout.EAST);

        card.add(top, BorderLayout.NORTH);
        card.add(controls, BorderLayout.SOUTH);
        return card;
    }

    private void showCard(String name) {
        cards.show(cardHost, name);
        boolean dashboard = "dashboard".equals(name);
        navDashboard.setBase(dashboard ? ACCENT : CARD_2);
        navAlerts.setBase(dashboard ? CARD_2 : ACCENT);
    }

    private JComponent buildDashboard() {
        JPanel wrap = new ScrollableColumn();
        wrap.setOpaque(false);
        wrap.setLayout(new BoxLayout(wrap, BoxLayout.Y_AXIS));
        wrap.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));

        banner.setAlignmentX(Component.LEFT_ALIGNMENT);
        banner.setMaximumSize(new Dimension(Integer.MAX_VALUE, 60));

        modeWarning.setForeground(MUTED);
        modeWarning.setFont(font(Font.PLAIN, 12f));
        modeWarning.setAlignmentX(Component.LEFT_ALIGNMENT);

        levelMeter.setAlignmentX(Component.LEFT_ALIGNMENT);
        levelMeter.setMaximumSize(new Dimension(Integer.MAX_VALUE, 28));

        diagResult.setForeground(MUTED);
        diagResult.setFont(font(Font.PLAIN, 12f));
        diagResult.setAlignmentX(Component.LEFT_ALIGNMENT);

        diagButton.setFont(font(Font.BOLD, 12f));
        diagButton.setAlignmentX(Component.LEFT_ALIGNMENT);
        diagButton.addActionListener(e -> runMicDiagnostic());

        wrap.add(banner);
        wrap.add(Box.createVerticalStrut(8));
        wrap.add(modeWarning);
        wrap.add(Box.createVerticalStrut(12));

        JPanel levelBody = transparent();
        levelBody.setLayout(new BoxLayout(levelBody, BoxLayout.Y_AXIS));
        levelMeter.setAlignmentX(Component.LEFT_ALIGNMENT);
        diagButton.setAlignmentX(Component.LEFT_ALIGNMENT);
        diagResult.setAlignmentX(Component.LEFT_ALIGNMENT);
        levelBody.add(levelMeter);
        levelBody.add(Box.createVerticalStrut(12));
        levelBody.add(diagButton);
        levelBody.add(Box.createVerticalStrut(6));
        levelBody.add(diagResult);
        wrap.add(card("INPUT LEVEL", levelBody));
        wrap.add(Box.createVerticalStrut(14));

        JPanel barsBody = transparent();
        barsBody.setLayout(new BoxLayout(barsBody, BoxLayout.Y_AXIS));
        for (CategoryRule rule : detectionConfig.categories) {
            ThreatCategory cat = rule.category();
            ScoreBar bar = new ScoreBar(CATEGORY_NAMES.get(cat), CATEGORY_COLORS.get(cat));
            bar.setAlignmentX(Component.LEFT_ALIGNMENT);
            bar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
            bar.setThreshold(rule.threshold());
            bars.put(cat, bar);
            barsBody.add(bar);
            barsBody.add(Box.createVerticalStrut(8));
        }
        barsBody.add(Box.createVerticalGlue());
        wrap.add(card("CATEGORY SCORES", barsBody));
        wrap.add(Box.createVerticalStrut(8));

        JScrollPane scroll = new JScrollPane(wrap);
        scroll.setBorder(null);
        scroll.getViewport().setBackground(BG);
        scroll.setOpaque(false);
        scroll.getVerticalScrollBar().setUnitIncrement(16);
        return scroll;
    }

    private JComponent buildAlertHistory() {
        JPanel panel = transparent(new BorderLayout(0, 12));

        alertTable.setBackground(CARD);
        alertTable.setForeground(TEXT);
        alertTable.setGridColor(CARD_2);
        alertTable.setRowHeight(30);
        alertTable.setShowVerticalLines(false);
        alertTable.setFont(font(Font.PLAIN, 13f));
        alertTable.setSelectionBackground(CARD_2);
        alertTable.setSelectionForeground(TEXT);
        alertTable.getTableHeader().setReorderingAllowed(false);
        alertTable.getTableHeader().setBackground(CARD_2);
        alertTable.getTableHeader().setForeground(MUTED);
        alertTable.getTableHeader().setFont(font(Font.BOLD, 12f));
        alertTable.setFillsViewportHeight(true);
        alertTable.getColumnModel().getColumn(1).setCellRenderer(new PriorityCellRenderer());
        alertTable.getColumnModel().getColumn(2).setCellRenderer(new CategoryCellRenderer());
        alertTable.getColumnModel().getColumn(1).setMaxWidth(44);
        alertTable.getColumnModel().getColumn(1).setMinWidth(44);

        JScrollPane scroll = new JScrollPane(alertTable);
        scroll.getViewport().setBackground(CARD);
        scroll.setBorder(BorderFactory.createLineBorder(BORDER));
        scroll.setOpaque(false);

        RoundedButton export = new RoundedButton("Export CSV...", CARD_2, 10);
        export.setFont(font(Font.BOLD, 12f));
        export.addActionListener(e -> exportAlerts());
        RoundedButton clear = new RoundedButton("Clear", CARD_2, 10);
        clear.setFont(font(Font.BOLD, 12f));
        clear.addActionListener(e -> clearAlerts());

        alertCount.setForeground(MUTED);
        alertCount.setFont(font(Font.BOLD, 12f));

        JPanel actions = transparent();
        actions.setLayout(new BoxLayout(actions, BoxLayout.X_AXIS));
        actions.add(export);
        actions.add(Box.createHorizontalStrut(8));
        actions.add(clear);

        JPanel south = transparent(new BorderLayout(8, 0));
        south.add(alertCount, BorderLayout.WEST);
        south.add(actions, BorderLayout.EAST);

        JPanel body = transparent(new BorderLayout(0, 12));
        body.add(scroll, BorderLayout.CENTER);
        body.add(south, BorderLayout.SOUTH);
        panel.add(card("ALERT HISTORY", body), BorderLayout.CENTER);
        return panel;
    }

    private JComponent buildStatusBar() {
        JPanel bar = transparent(new BorderLayout());
        status.setForeground(MUTED);
        status.setFont(font(Font.PLAIN, 11f));
        stats.setForeground(MUTED);
        stats.setFont(font(Font.PLAIN, 11f));
        stats.setHorizontalAlignment(SwingConstants.RIGHT);
        bar.add(status, BorderLayout.WEST);
        bar.add(stats, BorderLayout.EAST);
        return bar;
    }

    private static JPanel transparent() {
        JPanel p = new JPanel();
        p.setOpaque(false);
        return p;
    }

    private static JPanel transparent(LayoutManager layout) {
        JPanel p = new JPanel(layout);
        p.setOpaque(false);
        return p;
    }

    /** A rounded card with a small uppercase title and an optional subtitle. */
    private static JComponent card(String title, JComponent body) {
        RoundedPanel card = new RoundedPanel(new BorderLayout(0, 12), CARD, 14, BORDER);
        card.setBorder(BorderFactory.createEmptyBorder(16, 18, 16, 18));
        JLabel label = new JLabel(title);
        label.setForeground(MUTED);
        label.setFont(font(Font.BOLD, 11f));
        card.add(label, BorderLayout.NORTH);
        card.add(body, BorderLayout.CENTER);
        return card;
    }

    // ---- journal / priorities ----------------------------------------------    /** Local alert priority: P0 critical, P1 high, P4 informational. */
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
            case "P4" -> ACCENT;
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
        banner.setStatus(String.format("%s  \u26a0  %s   %.2f",
                priority(event.category()), name, event.score()), ALERT, Color.WHITE, Color.WHITE);
        bannerReset.restart();
    }

    private void showMonitoringBanner() {
        if (running) {
            banner.setStatus("MONITORING  \u00b7  " + prioritySummary(),
                    new Color(0x10251a), new Color(0x7ee2a8), OK);
        } else {
            showIdleBanner();
        }
    }

    private static String prioritySummary() {
        return "P0 gunshot/explosion  \u00b7  P1 scream/impact/glass  \u00b7  P4 siren";
    }

    private void showIdleBanner() {
        banner.setStatus("IDLE  \u00b7  press Start listening to monitor", CARD_2, MUTED, MUTED);
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
        startStop.setBase(isRunning ? ALERT : OK);
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
        javax.swing.JFileChooser chooser = new javax.swing.JFileChooser(".");
        chooser.setSelectedFile(new java.io.File("omniear-alerts.csv"));
        if (chooser.showSaveDialog(this) != javax.swing.JFileChooser.APPROVE_OPTION) {
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

    /** Transparent column that fills the scroll viewport width but not its height. */
    private static final class ScrollableColumn extends JPanel implements Scrollable {
        @Override
        public Dimension getPreferredScrollableViewportSize() {
            return getPreferredSize();
        }

        @Override
        public int getScrollableUnitIncrement(java.awt.Rectangle visibleRect, int orientation, int direction) {
            return 16;
        }

        @Override
        public int getScrollableBlockIncrement(java.awt.Rectangle visibleRect, int orientation, int direction) {
            return 80;
        }

        @Override
        public boolean getScrollableTracksViewportWidth() {
            return true;
        }

        @Override
        public boolean getScrollableTracksViewportHeight() {
            return false;
        }
    }

    /** Rounded, hover-aware button used for all actions and navigation pills. */
    private static final class RoundedButton extends JButton {
        private Color base;
        private final int radius;
        private boolean hover;

        RoundedButton(String text, Color base, int radius) {
            super(text);
            this.base = base;
            this.radius = radius;
            setContentAreaFilled(false);
            setFocusPainted(false);
            setOpaque(false);
            setForeground(Color.WHITE);
            setFont(font(Font.BOLD, 13f));
            setBorder(BorderFactory.createEmptyBorder(9, 18, 9, 18));
            setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseEntered(MouseEvent e) {
                    hover = true;
                    repaint();
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    hover = false;
                    repaint();
                }
            });
        }

        void setBase(Color base) {
            this.base = base;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            Color fill = isEnabled() ? base : base.darker();
            g2.setColor(hover && isEnabled() ? fill.brighter() : fill);
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), radius, radius);
            g2.dispose();
            super.paintComponent(g);
        }
    }

    /** Rounded panel with an optional border, used for cards and the header. */
    private static final class RoundedPanel extends JPanel {
        private final int radius;
        private final Color fill;
        private final Color border;

        RoundedPanel(LayoutManager layout, Color fill, int radius, Color border) {
            super(layout);
            this.radius = radius;
            this.fill = fill;
            this.border = border;
            setOpaque(false);
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(fill);
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), radius, radius);
            if (border != null) {
                g2.setColor(border);
                g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, radius, radius);
            }
            g2.dispose();
            super.paintComponent(g);
        }
    }

    /** Full-width rounded status banner with a leading status dot. */
    private static final class StatusBanner extends JComponent {
        private String text = "IDLE";
        private Color bg = CARD_2;
        private Color fg = MUTED;
        private Color dot = MUTED;

        StatusBanner() {
            setOpaque(false);
            setPreferredSize(new Dimension(100, 60));
        }

        void setStatus(String text, Color bg, Color fg, Color dot) {
            this.text = text;
            this.bg = bg;
            this.fg = fg;
            this.dot = dot;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            int w = getWidth();
            int h = getHeight();
            g2.setColor(bg);
            g2.fillRoundRect(0, 0, w, h, 14, 14);
            int r = 7;
            int cx = 22;
            g2.setColor(dot);
            g2.fillOval(cx, h / 2 - r, r * 2, r * 2);
            g2.setFont(font(Font.BOLD, 18f));
            g2.setColor(fg);
            FontMetrics fm = g2.getFontMetrics();
            int ty = (h - fm.getHeight()) / 2 + fm.getAscent();
            g2.drawString(text, cx + r * 2 + 14, ty);
            g2.dispose();
        }
    }

    private static final class LevelMeter extends JComponent {
        private volatile float level;
        private volatile float peak;

        LevelMeter() {
            setOpaque(false);
            setPreferredSize(new Dimension(100, 26));
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
            g2.setColor(CARD_2);
            g2.fillRoundRect(0, 0, w, h, h, h);
            int fill = (int) (level * (w - 2));
            if (fill > 0) {
                g2.setPaint(new GradientPaint(1, 0, levelColor(level).darker(), Math.max(fill, 2), 0,
                        levelColor(level)));
                g2.fillRoundRect(1, 1, fill, h - 2, h - 2, h - 2);
            }
            int px = 1 + (int) (peak * (w - 2));
            g2.setColor(Color.WHITE);
            g2.fillRoundRect(Math.min(px, w - 3), 1, 3, h - 2, 2, 2);
            g2.dispose();
        }

        private static Color levelColor(float level) {
            if (level > 0.85f) {
                return ALERT;
            }
            if (level > 0.6f) {
                return WARN;
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
            g2.setColor(CARD_2);
            g2.fillRoundRect(0, 0, w, h, 9, 9);

            boolean hot = score >= threshold;
            int fill = (int) (Math.min(score, 1f) * (w - 2));
            if (fill > 0) {
                Color c = hot ? color : color.darker().darker();
                g2.setPaint(new GradientPaint(1, 0, c.darker(), Math.max(fill, 2), 0, c.brighter()));
                g2.fillRoundRect(1, 1, fill, h - 2, 8, 8);
            }

            int tick = (int) (Math.min(threshold, 1f) * (w - 2));
            g2.setColor(new Color(0xffffff));
            g2.fillRoundRect(tick + 1, 4, 2, h - 8, 2, 2);

            g2.setFont(font(Font.BOLD, 12f));
            g2.setColor(hot ? Color.WHITE : TEXT);
            FontMetrics fm = g2.getFontMetrics();
            int ty = (h - fm.getHeight()) / 2 + fm.getAscent();
            g2.drawString(name, 12, ty);
            String value = String.format("%.2f", score);
            g2.drawString(value, w - fm.stringWidth(value) - 12, ty);
            g2.dispose();
        }
    }

    private static final class CategoryCellRenderer extends DefaultTableCellRenderer {
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                       boolean hasFocus, int row, int column) {
            Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
            c.setBackground(isSelected ? CARD_2 : CARD);
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
            c.setBackground(isSelected ? CARD_2 : CARD);
            setHorizontalAlignment(SwingConstants.CENTER);
            setFont(font(Font.BOLD, 12f));
            c.setForeground(value == null ? MUTED : priorityColor(value.toString()));
            return c;
        }
    }
}
