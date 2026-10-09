package com.example.omniear.audio;

import java.io.IOException;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads from an {@link AudioSource}, converts to mono float32 at the model rate,
 * keeps the rolling buffer, and emits overlapping {@link AudioWindow}s on a
 * background thread. Capture never blocks on inference; the sink is expected to
 * hand windows to a bounded queue and drop stale ones.
 */
public final class AudioCaptureService implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(AudioCaptureService.class);
    private static final int READ_BYTES = 4096;

    private final AudioSource source;
    private final AudioFormatAdapter adapter;
    private final RollingAudioBuffer buffer;
    private final int modelRate;
    private final int windowSamples;
    private final int hopSamples;
    private final Consumer<AudioWindow> windowSink;
    private final Runnable onEndOfStream;

    private volatile boolean running;
    private Thread thread;
    private long nextWindowAt;

    public AudioCaptureService(AudioSource source,
                               int modelRate,
                               int windowSamples,
                               int hopSamples,
                               Consumer<AudioWindow> windowSink,
                               Runnable onEndOfStream) {
        this.source = source;
        this.adapter = new AudioFormatAdapter(source.format(), modelRate);
        this.buffer = new RollingAudioBuffer(windowSamples);
        this.modelRate = modelRate;
        this.windowSamples = windowSamples;
        this.hopSamples = hopSamples;
        this.windowSink = windowSink;
        this.onEndOfStream = onEndOfStream;
        this.nextWindowAt = windowSamples;
    }

    public void start() {
        running = true;
        thread = new Thread(this::loop, "audio-capture");
        thread.setDaemon(true);
        thread.start();
    }

    private void loop() {
        byte[] raw = new byte[READ_BYTES];
        try {
            int n;
            while (running && (n = source.read(raw, 0, raw.length)) != -1) {
                float[] chunk = adapter.convert(raw, n);
                if (chunk.length == 0) {
                    continue;
                }
                buffer.add(chunk);
                long total = buffer.totalWritten();
                if (total >= nextWindowAt) {
                    long startSample = nextWindowAt - windowSamples;
                    float[] samples = buffer.lastWindow(windowSamples);
                    long audioEndMs = (nextWindowAt * 1000L) / modelRate;
                    windowSink.accept(new AudioWindow(
                            samples, startSample, audioEndMs, System.currentTimeMillis()));
                    // Advance and skip stale boundaries if capture outpaced inference.
                    do {
                        nextWindowAt += hopSamples;
                    } while (nextWindowAt <= total);
                }
            }
        } catch (IOException e) {
            if (running) {
                LOG.error("audio capture failed", e);
            }
        } finally {
            running = false;
            if (onEndOfStream != null) {
                onEndOfStream.run();
            }
        }
    }

    public boolean isRunning() {
        return running;
    }

    public long samplesCaptured() {
        return buffer.totalWritten();
    }

    public void stop() {
        running = false;
    }

    @Override
    public void close() {
        stop();
        try {
            source.close();
        } finally {
            if (thread != null) {
                try {
                    thread.join(2000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }
}
