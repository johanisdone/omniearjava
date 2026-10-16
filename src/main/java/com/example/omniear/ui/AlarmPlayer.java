package com.example.omniear.ui;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Plays short synthesized alarm tones on a background thread, so the Swing event
 * thread and the capture pipeline are never blocked. Tones are generated as raw
 * 16-bit PCM and streamed to a {@link SourceDataLine}; no audio files are needed.
 */
final class AlarmPlayer {

    private static final Logger LOG = LoggerFactory.getLogger(AlarmPlayer.class);
    private static final float SAMPLE_RATE = 44100f;
    private static final AudioFormat FORMAT = new AudioFormat(SAMPLE_RATE, 16, 1, true, false);
    /** Fade in/out length in samples, to avoid clicks at segment edges. */
    private static final int FADE = 64;

    private final ExecutorService exec = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "omniear-alarm");
        t.setDaemon(true);
        return t;
    });

    /** Queues one alarm burst for the given priority (played, then discarded). */
    void play(AlertPriority priority) {
        exec.submit(() -> {
            try (SourceDataLine line = AudioSystem.getSourceDataLine(FORMAT)) {
                line.open(FORMAT);
                line.start();
                byte[] pcm = render(priority.tones());
                line.write(pcm, 0, pcm.length);
                line.drain();
                line.stop();
            } catch (Exception e) {
                LOG.warn("could not play {} alarm: {}", priority.label(), e.toString());
            }
        });
    }

    void shutdown() {
        exec.shutdownNow();
    }

    private static byte[] render(List<AlertPriority.Tone> tones) {
        int total = 0;
        for (AlertPriority.Tone t : tones) {
            total += samples(t.ms());
        }
        byte[] out = new byte[total * 2];
        int idx = 0;
        for (AlertPriority.Tone t : tones) {
            int n = samples(t.ms());
            for (int i = 0; i < n; i++) {
                short s = 0;
                if (t.hz() > 0) {
                    double wave = Math.sin(2 * Math.PI * t.hz() * i / SAMPLE_RATE);
                    s = (short) (wave * envelope(i, n) * t.amp() * Short.MAX_VALUE);
                }
                out[idx++] = (byte) (s & 0xff);
                out[idx++] = (byte) ((s >> 8) & 0xff);
            }
        }
        return out;
    }

    private static int samples(int ms) {
        return Math.max(1, Math.round(ms * SAMPLE_RATE / 1000f));
    }

    private static float envelope(int i, int n) {
        int fade = Math.min(FADE, Math.max(1, n / 4));
        if (i < fade) {
            return i / (float) fade;
        }
        if (i > n - fade) {
            return (n - i) / (float) fade;
        }
        return 1f;
    }
}
