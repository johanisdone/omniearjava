package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.example.omniear.model.AudioPreprocessor;
import com.example.omniear.model.ModelConfig;
import org.junit.jupiter.api.Test;

class AudioPreprocessorTest {

    @Test
    void windowOffsetsUseFloorStepping() {
        // 2s window, 1s hop, 3s signal -> offsets {0, 16000}
        assertArrayEquals(new int[]{0, 16000},
                AudioPreprocessor.windowOffsets(48000, 32000, 16000));
    }

    @Test
    void shortSignalYieldsSinglePaddedWindow() {
        assertArrayEquals(new int[]{0},
                AudioPreprocessor.windowOffsets(100, 32000, 16000));
    }

    @Test
    void windowZeroPadsShortInput() {
        float[] samples = {1f, 2f, 3f};
        float[] w = AudioPreprocessor.window(samples, 0, 5);
        assertArrayEquals(new float[]{1f, 2f, 3f, 0f, 0f}, w);
    }

    @Test
    void linearResampleProducesExpectedLength() {
        float[] in = new float[100];
        for (int i = 0; i < in.length; i++) {
            in[i] = i;
        }
        float[] out = AudioPreprocessor.resampleLinear(in, 8000, 16000);
        assertEquals(200, out.length);
        assertEquals(0f, out[0], 1e-6);
    }

    @Test
    void normalizePeakAmplifiesQuietWindowTowardTarget() {
        float[] in = {0.04f, -0.02f};
        float[] out = AudioPreprocessor.normalizePeak(in);
        assertArrayEquals(new float[]{0.9f, -0.45f}, out, 1e-6f);
    }

    @Test
    void normalizePeakLeavesBelowFloorWindowUntouched() {
        float[] in = {0.005f, -0.003f};
        assertSame(in, AudioPreprocessor.normalizePeak(in));
    }

    @Test
    void normalizePeakLeavesNearFullScaleWindowUntouched() {
        float[] in = {0.95f, -0.9f};
        assertSame(in, AudioPreprocessor.normalizePeak(in));
    }

    @Test
    void normalizePeakCapsGainAtMax() {
        float[] out = AudioPreprocessor.normalizePeak(new float[]{0.02f}, 0.02f, 0.9f, 10f);
        assertArrayEquals(new float[]{0.2f}, out, 1e-6f);
    }

    @Test
    void configWindowAndHopSamples() {
        ModelConfig cfg = new ModelConfig();
        cfg.sr = 16000;
        cfg.win_s = 2.0;
        cfg.hop_s = 1.0;
        assertEquals(32000, cfg.windowSamples());
        assertEquals(16000, cfg.hopSamples());
    }
}
