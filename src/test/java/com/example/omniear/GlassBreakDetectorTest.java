package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.example.omniear.detection.GlassBreakDetector;
import java.util.Random;
import org.junit.jupiter.api.Test;

class GlassBreakDetectorTest {

    private final GlassBreakDetector detector = new GlassBreakDetector();

    @Test
    void detectsHighFrequencyNoiseBurst() {
        float[] x = new float[32000];
        Random r = new Random(42);
        float prev = 0f;
        for (int i = 16000; i < 17600; i++) {
            float n = r.nextFloat() * 2 - 1;
            x[i] = n - prev; // first-order high-pass -> high-frequency dominated burst
            prev = n;
        }
        assertTrue(detector.detects(x));
    }

    @Test
    void ignoresLowFrequencyTone() {
        float[] x = new float[32000];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (0.5 * Math.sin(2 * Math.PI * 200 * i / 16000));
        }
        assertFalse(detector.detects(x));
    }

    @Test
    void ignoresSilence() {
        assertFalse(detector.detects(new float[32000]));
    }
}
