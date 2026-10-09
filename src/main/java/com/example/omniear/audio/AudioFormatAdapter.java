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
    private double carry;
    private float prev;

    public AudioFormatAdapter(AudioFormat source, int targetRate) {
        if (source.getSampleSizeInBits() != 16) {
            throw new IllegalArgumentException("adapter requires 16-bit PCM, got " + source);
        }
        this.sourceRate = (int) Math.round(source.getSampleRate());
        this.channels = Math.max(1, source.getChannels());
        this.targetRate = targetRate;
        this.step = sourceRate / (double) targetRate;
        this.carry = 0.0;
        this.prev = 0f;
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
        // Conceptual input stream: [prev, in[0], in[1], ...]
        int n = in.length;
        float[] work = new float[n + 1];
        work[0] = prev;
        System.arraycopy(in, 0, work, 1, n);

        double pos = carry;
        List<Float> out = new ArrayList<>(Math.max(1, (int) (n / step) + 2));
        while (pos < work.length - 1) {
            int i = (int) pos;
            double frac = pos - i;
            out.add((float) (work[i] * (1 - frac) + work[i + 1] * frac));
            pos += step;
        }
        carry = pos - (work.length - 1);
        if (carry < 0) {
            carry = 0;
        }
        prev = work[work.length - 1];

        float[] result = new float[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return result;
    }
}
