package com.example.omniear.audio;

import java.util.ArrayList;
import java.util.List;
import javax.sound.sampled.AudioFormat;

/**
 * Converts captured 16-bit PCM bytes into mono float32 at the model sample rate.
 *
 * <p>Handles channel mixing and (when the device cannot deliver the target rate)
 * a stateful linear resampler that carries fractional phase across chunks.
 * Assumes 16-bit signed little-endian input, which {@link MicrophoneSource} and
 * {@link FileAudioSource} guarantee.
 */
public final class AudioFormatAdapter {

    private final int sourceRate;
    private final int channels;
    private final int targetRate;
    private final double step; // input samples per output sample

    // Streaming linear-resampler state. Mirrors AudioPreprocessor.resampleLinear so
    // live capture produces the same samples the model was trained and validated on.
    private float[] history = new float[8192];
    private int historyLen;
    private long historyStart; // global index of history[0]
    private double pos;        // global position of the next output sample

    public AudioFormatAdapter(AudioFormat source, int targetRate) {
        if (source.getSampleSizeInBits() != 16) {
            throw new IllegalArgumentException("adapter requires 16-bit PCM, got " + source);
        }
        this.sourceRate = (int) Math.round(source.getSampleRate());
        this.channels = Math.max(1, source.getChannels());
        this.targetRate = targetRate;
        this.step = sourceRate / (double) targetRate;
        this.historyStart = 0L;
        this.pos = 0.0;
    }

    public int targetRate() {
        return targetRate;
    }

    /** Convert {@code len} bytes of PCM into mono float32 samples at the target rate. */
    public float[] convert(byte[] buffer, int len) {
        int frameSize = channels * 2;
        int frames = len / frameSize;
        if (frames == 0) {
            return new float[0];
        }
        float[] mono = new float[frames];
        for (int f = 0; f < frames; f++) {
            float sum = 0f;
            for (int c = 0; c < channels; c++) {
                int lo = buffer[f * frameSize + c * 2] & 0xFF;
                int hi = buffer[f * frameSize + c * 2 + 1];
                sum += ((hi << 8) | lo) / 32768f;
            }
            mono[f] = sum / channels;
        }
        if (sourceRate == targetRate || step <= 0) {
            return mono;
        }
        return resample(mono);
    }

    private float[] resample(float[] in) {
        if (historyLen + in.length > history.length) {
            int cap = Math.max(historyLen + in.length, history.length * 2);
            float[] grown = new float[cap];
            System.arraycopy(history, 0, grown, 0, historyLen);
            history = grown;
        }
        System.arraycopy(in, 0, history, historyLen, in.length);
        historyLen += in.length;

        List<Float> out = new ArrayList<>(Math.max(1, (int) (in.length / step) + 2));
        long lastGlobal = historyStart + historyLen - 1;
        while (true) {
            long i0g = (long) Math.floor(pos);
            if (i0g + 1 > lastGlobal) {
                break;
            }
            int i0 = (int) (i0g - historyStart);
            double frac = pos - i0g;
            out.add((float) (history[i0] * (1 - frac) + history[i0 + 1] * frac));
            pos += step;
        }

        // Drop samples strictly before the next interpolation anchor, keeping one.
        long anchor = (long) Math.floor(pos);
        int drop = (int) Math.min(anchor - historyStart, historyLen - 1);
        if (drop > 0) {
            System.arraycopy(history, drop, history, 0, historyLen - drop);
            historyLen -= drop;
            historyStart += drop;
        }

        float[] result = new float[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return result;
    }
}
