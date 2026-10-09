package com.example.omniear.ui;

import com.example.omniear.audio.AudioCaptureService;
import com.example.omniear.audio.AudioWindow;
import com.example.omniear.audio.MicrophoneSource;
import com.example.omniear.detection.CategoryRule;
import com.example.omniear.detection.DetectionConfig;
import com.example.omniear.detection.MappedScores;
import com.example.omniear.detection.ScoreMapper;
import com.example.omniear.detection.TemporalDecisionEngine;
import com.example.omniear.detection.ThreatCategory;
import com.example.omniear.detection.ThreatEvent;
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
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
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
import javax.swing.BoxLayout;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;

/**
 * Live microphone dashboard for the OmniEar pipeline: device picker, input-level
 * meter, per-category score bars with threshold markers, and a rolling alert log.
 *
 * <p>The model, decision engine and capture run on background threads; every
 * update is marshalled onto the Swing event thread.
 */
public final class OmniEarFrame extends JFrame {

    private static final Color BG = new Color(0x16161c);
    private static final Color PANEL = new Color(0x1f1f28);
    private static final Color PANEL_2 = new Color(0x272733);
    private static final Color TEXT = new Color(0xe6e6ee);
    private static final Color MUTED = new Color(0x9a9aad);
    private static final Color OK = new Color(0x2fbf71);
    private static final Color ALERT = new Color(0xe5484d);
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

    private final ModelConfig modelConfig;
    private final DetectionConfig detectionConfig;
    private final Path modelsDir;
    private final Map<ThreatCategory, Float> thresholds = new EnumMap<>(ThreatCategory.class);

    private final JComboBox<String> deviceCombo = new JComboBox<>();
    private final JButton startStop = new JButton("Start listening");
    private final JLabel banner = new JLabel("IDLE", SwingConstants.CENTER);
    private final JLabel status = new JLabel("Ready");
    private final JLabel stats = new JLabel(" ");
    private final LevelMeter levelMeter = new LevelMeter();
    private final Map<ThreatCategory, ScoreBar> bars = new EnumMap<>(ThreatCategory.class);
    private final DefaultTableModel alertModel = new DefaultTableModel(
            new Object[] {"Time", "Category", "Score", "Dur", "Label"}, 0) {
        @Override
        public boolean isCellEditable(int row, int column) {
            return false;
        }
    };
    private final JTable alertTable = new JTable(alertModel);

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

    private void buildUi() {
        JPanel root = new JPanel(new BorderLayout(12, 12));
        root.setBackground(BG);
        root.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));
        root.add(buildToolbar(), BorderLayout.NORTH);
        root.add(buildCenter(), BorderLayout.CENTER);
        root.add(buildStatusBar(), BorderLayout.SOUTH);
        setContentPane(root);
        setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                shutdown();
            }
        });
        setSize(720, 720);
        setMinimumSize(new Dimension(560, 560));
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

        DefaultComboBoxModel<String> model = new DefaultComboBoxModel<>();
        model.addElement("<System default>");
        for (String name : MicrophoneSource.captureDeviceNames()) {
            model.addElement(name);
        }
        deviceCombo.setModel(model);
        deviceCombo.setPreferredSize(new Dimension(320, 30));
        deviceCombo.setBackground(PANEL_2);
        deviceCombo.setForeground(TEXT);

        JPanel left = new JPanel();
        left.setOpaque(false);
        left.setLayout(new BoxLayout(left, BoxLayout.X_AXIS));
        left.add(deviceLabel);
        left.add(javax.swing.Box.createHorizontalStrut(8));
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

    private JComponent buildCenter() {
        JPanel center = new JPanel(new BorderLayout(12, 12));
        center.setOpaque(false);
        center.add(buildMeters(), BorderLayout.NORTH);
        center.add(buildAlerts(), BorderLayout.CENTER);
        return center;
    }

    private JComponent buildMeters() {
        JPanel wrap = new JPanel();
        wrap.setOpaque(false);
        wrap.setLayout(new BoxLayout(wrap, BoxLayout.Y_AXIS));

        banner.setOpaque(true);
        banner.setFont(banner.getFont().deriveFont(Font.BOLD, 22f));
        banner.setBorder(BorderFactory.createEmptyBorder(16, 12, 16, 12));
        banner.setAlignmentX(Component.LEFT_ALIGNMENT);
        banner.setMaximumSize(new Dimension(Integer.MAX_VALUE, 64));

        JLabel levelLabel = sectionLabel("INPUT LEVEL");
        levelMeter.setAlignmentX(Component.LEFT_ALIGNMENT);
        levelMeter.setMaximumSize(new Dimension(Integer.MAX_VALUE, 26));

        wrap.add(banner);
        wrap.add(javax.swing.Box.createVerticalStrut(12));
        wrap.add(levelLabel);
        wrap.add(javax.swing.Box.createVerticalStrut(6));
        wrap.add(levelMeter);
        wrap.add(javax.swing.Box.createVerticalStrut(14));
        wrap.add(sectionLabel("CATEGORY SCORES"));
        wrap.add(javax.swing.Box.createVerticalStrut(6));

        for (CategoryRule rule : detectionConfig.categories) {
            ThreatCategory cat = rule.category();
            ScoreBar bar = new ScoreBar(CATEGORY_NAMES.get(cat), CATEGORY_COLORS.get(cat));
            bar.setAlignmentX(Component.LEFT_ALIGNMENT);
            bar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
            bar.setThreshold(rule.threshold());
            bars.put(cat, bar);
            wrap.add(bar);
            wrap.add(javax.swing.Box.createVerticalStrut(6));
        }
        return wrap;
    }

    private JComponent buildAlerts() {
        JPanel panel = new JPanel(new BorderLayout(0, 6));
        panel.setOpaque(false);
        panel.setBorder(BorderFactory.createEmptyBorder(6, 0, 0, 0));
        panel.add(sectionLabel("ALERT LOG"), BorderLayout.NORTH);

        alertTable.setBackground(PANEL);
        alertTable.setForeground(TEXT);
        alertTable.setGridColor(PANEL_2);
        alertTable.setRowHeight(26);
        alertTable.setFont(alertTable.getFont().deriveFont(Font.PLAIN, 13f));
        alertTable.getTableHeader().setReorderingAllowed(false);
        alertTable.setFillsViewportHeight(true);
        alertTable.getColumnModel().getColumn(1).setCellRenderer(new CategoryCellRenderer());

        JScrollPane scroll = new JScrollPane(alertTable);
        scroll.getViewport().setBackground(PANEL);
        scroll.setBorder(BorderFactory.createLineBorder(PANEL_2));
        panel.add(scroll, BorderLayout.CENTER);
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

    private static JLabel sectionLabel(String text) {
        JLabel label = new JLabel(text);
        label.setForeground(MUTED);
        label.setFont(label.getFont().deriveFont(Font.BOLD, 11f));
        label.setAlignmentX(Component.LEFT_ALIGNMENT);
        return label;
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
                () -> SwingUtilities.invokeLater(() -> onCaptureEnded()));

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
            ModelPrediction prediction = model.predict(window.samples());
            MappedScores mapped = mapper.map(prediction);
            List<ThreatEvent> events = engine.update(window.timestampMs(), mapped);
            windows.incrementAndGet();

            float db = (float) (20 * Math.log10(Math.max(rms(window.samples()), 1e-6)));
            float peakDb = (float) (20 * Math.log10(Math.max(peak(window.samples()), 1e-6)));
            float level = normalizeDb(db);
            float peakLevel = normalizeDb(peakDb);
            SwingUtilities.invokeLater(() -> updateLive(level, peakLevel, mapped, events));
        }
    }

    private void updateLive(float level, float peakLevel, MappedScores mapped, List<ThreatEvent> events) {
        levelMeter.setLevels(level, peakLevel);
        for (Map.Entry<ThreatCategory, ScoreBar> e : bars.entrySet()) {
            ThreatCategory cat = e.getKey();
            float score = mapped.score(cat);
            e.getValue().setScore(score);
        }
        for (ThreatEvent event : events) {
            alerts.incrementAndGet();
            addAlertRow(event);
            flashAlert(event);
        }
        stats.setText(String.format("windows=%d  dropped=%d  alerts=%d",
                windows.get(), dropped.get(), alerts.get()));
    }

    private void addAlertRow(ThreatEvent event) {
        String time = Instant.ofEpochMilli(event.emittedWallMs())
                .atZone(ZoneId.systemDefault()).toLocalTime().format(TIME);
        alertModel.insertRow(0, new Object[] {
                time,
                CATEGORY_NAMES.getOrDefault(event.category(), event.category().name()),
                String.format("%.2f", event.score()),
                event.durationMs() + " ms",
                event.drivingLabel()});
        while (alertModel.getRowCount() > 300) {
            alertModel.removeRow(alertModel.getRowCount() - 1);
        }
    }

    private void flashAlert(ThreatEvent event) {
        String name = CATEGORY_NAMES.getOrDefault(event.category(), event.category().name());
        banner.setText(String.format("\u26a0  %s   %.2f", name, event.score()));
        banner.setBackground(ALERT);
        banner.setForeground(Color.WHITE);
        bannerReset.restart();
    }

    private void showMonitoringBanner() {
        if (running) {
            banner.setText("\u25cf  MONITORING");
            banner.setBackground(new Color(0x15351f));
            banner.setForeground(new Color(0x7ee2a8));
        } else {
            showIdleBanner();
        }
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

    private void onCaptureEnded() {
        if (running) {
            stopPipeline("Capture ended");
        }
    }

    private void fail(String message) {
        startStop.setEnabled(true);
        status.setText(message);
        showIdleBanner();
        javax.swing.JOptionPane.showMessageDialog(this, message, "OmniEar", javax.swing.JOptionPane.ERROR_MESSAGE);
    }

    private void setRunningUi(boolean isRunning) {
        startStop.setEnabled(true);
        startStop.setText(isRunning ? "Stop" : "Start listening");
        startStop.setBackground(isRunning ? ALERT : OK);
        deviceCombo.setEnabled(!isRunning);
        if (isRunning) {
            showMonitoringBanner();
        }
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
}
