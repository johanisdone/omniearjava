package com.example.omniear.detection;

import java.util.Arrays;

/**
 * Acoustic glass-break detector, used <em>in addition to</em> the ML model.
 *
 * <p>The CED-mini backbone recognizes full-band glass breaks well, but
 * band-limited renditions (e.g. a phone speaker) strip the low-frequency body of
 * the sound, leaving only a high-frequency transient that the model no longer
 * maps to its "glass" class. This detector keys on that residual signature: an
 * impulsive ({@code crest}), high-frequency-dominant ({@code hfRatio}),
 * noise-like ({@code flatness}) transient.
 *
 * <p>All three features are ratios and therefore level-invariant, so the detector
 * can run on the raw window before normalization. It is intended to fire the
 * {@code GLASS_BREAK} category in addition to the model score.
 */
public final class GlassBreakDetector {

    private static final int SR = 16000;
    private static final int FRAME = 512;   // 32 ms analysis frame
    private static final int HOP = 160;     // 10 ms
    private static final float HF_CUTOFF = 2000f;
    private static final float HF_RATIO_MIN = 0.80f;
    private static final float CREST_MIN = 20000f;
    private static final float FLATNESS_MIN = 0.30f;

    public boolean detects(float[] samples) {
        if (samples.length < FRAME) {
            return false;
        }
        float[] hp = highPass(samples);
        int frames = 1 + (samples.length - FRAME) / HOP;
        double[] hfEnergy = new double[frames];
        double[] fullEnergy = new double[frames];
        for (int i = 0; i < frames; i++) {
            int o = i * HOP;
            double eh = 0;
            double ef = 0;
            for (int k = 0; k < FRAME; k++) {
                double h = hp[o + k];
                double v = samples[o + k];
                eh += h * h;
                ef += v * v;
            }
            hfEnergy[i] = eh + 1e-12;
            fullEnergy[i] = ef + 1e-12;
        }

        int peak = 0;
        for (int i = 1; i < frames; i++) {
            if (hfEnergy[i] > hfEnergy[peak]) {
                peak = i;
            }
        }
        double[] sorted = hfEnergy.clone();
        Arrays.sort(sorted);
        double floor = sorted[frames / 2] + 1e-12;

        double crest = hfEnergy[peak] / floor;
        double hfRatio = hfEnergy[peak] / fullEnergy[peak];
        if (hfRatio < HF_RATIO_MIN || crest < CREST_MIN) {
            return false;
        }
        return flatness(hp, peak * HOP) >= FLATNESS_MIN;
    }

    /** Second-order Butterworth high-pass, applied to the whole window. */
    private float[] highPass(float[] x) {
        double w0 = 2 * Math.PI * HF_CUTOFF / SR;
        double cosw = Math.cos(w0);
        double alpha = Math.sin(w0) / Math.sqrt(2); // Q = 1/sqrt(2)
        double a0 = 1 + alpha;
        double b0 = (1 + cosw) / 2 / a0;
        double b1 = -(1 + cosw) / a0;
        double b2 = (1 + cosw) / 2 / a0;
        double a1 = -2 * cosw / a0;
        double a2 = (1 - alpha) / a0;

        float[] y = new float[x.length];
        double x1 = 0;
        double x2 = 0;
        double y1 = 0;
        double y2 = 0;
        for (int i = 0; i < x.length; i++) {
            double v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            y[i] = (float) v;
            x2 = x1;
            x1 = x[i];
            y2 = y1;
            y1 = v;
        }
        return y;
    }

    /** Geometric/arithmetic mean ratio of the 2-8 kHz magnitude spectrum of one frame. */
    private double flatness(float[] hp, int offset) {
        int n = FRAME;
        double[] window = new double[n];
        for (int i = 0; i < n; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
        }
        int lo = (int) Math.ceil(HF_CUTOFF * n / SR);
        int hi = (int) Math.floor(8000.0 * n / SR);
        double logSum = 0;
        double linSum = 0;
        int count = 0;
        for (int k = lo; k < hi; k++) {
            double re = 0;
            double im = 0;
            for (int i = 0; i < n; i++) {
                double s = hp[offset + i] * window[i];
                double ang = -2 * Math.PI * k * i / n;
                re += s * Math.cos(ang);
                im += s * Math.sin(ang);
            }
            double mag = Math.sqrt(re * re + im * im) + 1e-12;
            logSum += Math.log(mag);
            linSum += mag;
            count++;
        }
        if (count == 0) {
            return 0;
        }
        double geo = Math.exp(logSum / count);
        double arith = linSum / count;
        return geo / arith;
    }
}
