package com.example.omniear.model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Converts captured audio to the exact model-required representation:
 * mono, 16-bit-scaled float32 in [-1, 1], at the model sample rate.
 *
 * <p>The model window/hop come from {@link ModelConfig}; this class never
 * applies log-mel or any spectral transform because the ONNX backbone performs
 * the STFT/mel frontend internally from the raw waveform.
 */
public final class AudioPreprocessor {

    private AudioPreprocessor() {
    }

    /** Read a WAV/audio file into mono float32 samples at {@code targetRate}. */
    public static float[] readMono(Path file, int targetRate)
            throws IOException, UnsupportedAudioFileException {
        try (AudioInputStream src = AudioSystem.getAudioInputStream(file.toFile())) {
            return decode(src, targetRate);
        }
    }

    private static float[] decode(AudioInputStream src, int targetRate)
            throws IOException {
        AudioFormat source = src.getFormat();
        int channels = Math.max(1, source.getChannels());
        int frameSize = Math.max(1, channels * 2);

        // Normalize everything to PCM_SIGNED 16-bit little-endian first.
        AudioFormat pcm16 = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                source.getSampleRate(), 16, channels, frameSize, source.getSampleRate(), false);
        AudioInputStream stream = src;
        if (!source.getEncoding().equals(AudioFormat.Encoding.PCM_SIGNED)
                || source.getSampleSizeInBits() != 16) {
            stream = AudioSystem.getAudioInputStream(pcm16, src);
        }
        byte[] bytes = stream.readAllBytes();
        if (stream != src) {
            stream.close();
        }

        int nFrames = bytes.length / frameSize;
        float[] mono = new float[nFrames];
        for (int f = 0; f < nFrames; f++) {
            float sum = 0f;
            for (int c = 0; c < channels; c++) {
                int lo = (bytes[f * frameSize + c * 2] & 0xFF);
                int hi = bytes[f * frameSize + c * 2 + 1]; // signed high byte
                int sample = (hi << 8) | lo;
                sum += sample / 32768f;
            }
            mono[f] = sum / channels;
        }

        float sampleRate = source.getSampleRate();
        if (Math.abs(sampleRate - targetRate) > 0.5f) {
            return resampleLinear(mono, sampleRate, targetRate);
        }
        return mono;
    }

    /** Linear-interpolation resampler. Adequate for speech/event audio capture. */
    public static float[] resampleLinear(float[] x, float sourceRate, int targetRate) {
        if (x.length == 0 || Math.abs(sourceRate - targetRate) < 0.5f) {
            return x;
        }
        double ratio = targetRate / (double) sourceRate;
        int outLen = (int) Math.floor(x.length * ratio);
        float[] out = new float[outLen];
        for (int i = 0; i < outLen; i++) {
            double pos = i / ratio;
            int i0 = (int) Math.floor(pos);
            int i1 = Math.min(i0 + 1, x.length - 1);
            double frac = pos - i0;
            out[i] = (float) (x[i0] * (1 - frac) + x[i1] * frac);
        }
        return out;
    }

    /**
     * Window offsets over {@code length} samples with the given window/hop,
     * mirroring the reference pipeline: floor-stepped windows when at least one
     * full window fits, otherwise a single zero-padded window at offset 0.
     */
    public static int[] windowOffsets(int length, int window, int hop) {
        if (length < window) {
            return new int[]{0};
        }
        List<Integer> offs = new ArrayList<>();
        for (int o = 0; o + window <= length; o += hop) {
            offs.add(o);
        }
        int[] a = new int[offs.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = offs.get(i);
        }
        return a;
    }

    /**
     * Peak-normalize a window toward {@code target} before it reaches the model.
     *
     * <p>CED-mini expects audio close to full scale; laptop/phone-microphone chains
     * often deliver events at a small fraction of full scale (measured live here at
     * peak ~0.08), which the model misclassifies (gunshots heard as generic impacts).
     * Windows quieter than {@code floor} are left untouched so silence and low-level
     * room noise are not amplified into hallucinations. Gain is capped at
     * {@code maxGain}.
     */
    public static float[] normalizePeak(float[] window) {
        return normalizePeak(window, 0.02f, 0.9f, 30f);
    }

    public static float[] normalizePeak(float[] window, float floor, float target, float maxGain) {
        float peak = 0f;
        for (float v : window) {
            peak = Math.max(peak, Math.abs(v));
        }
        if (peak < floor) {
            return window;
        }
        float gain = target / peak;
        if (gain > maxGain) {
            gain = maxGain;
        }
        if (gain <= 1.001f) {
            return window;
        }
        float[] out = new float[window.length];
        for (int i = 0; i < window.length; i++) {
            out[i] = Math.max(-1f, Math.min(1f, window[i] * gain));
        }
        return out;
    }

    /** Extract a single window of exactly {@code window} samples, zero-padded. */
    public static float[] window(float[] samples, int offset, int window) {
        float[] out = new float[window];
        int available = Math.min(window,Math.max(0, samples.length - offset));
        if (available > 0) {
            System.arraycopy(samples, offset, out, 0, available);
        }
        return out;
    }

    /** All overlapping windows for a signal, per {@link ModelConfig}. */
    public static float[][] windows(float[] samples, ModelConfig config) {
        int win = config.windowSamples();
        int hop = config.hopSamples();
        int[] offs = windowOffsets(samples.length, win, hop);
        float[][] out = new float[offs.length][];
        for (int i = 0; i < offs.length; i++) {
            out[i] = window(samples, offs[i], win);
        }
        return out;
    }
}
