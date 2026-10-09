package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.example.omniear.audio.RollingAudioBuffer;
import org.junit.jupiter.api.Test;

class RollingAudioBufferTest {

    @Test
    void zeroPadsBeforeStreamStart() {
        RollingAudioBuffer b = new RollingAudioBuffer(4);
        b.add(new float[]{1, 2, 3});
        assertArrayEquals(new float[]{0, 1, 2, 3}, b.lastWindow(4));
    }

    @Test
    void returnsMostRecentSamplesAcrossWraparound() {
        RollingAudioBuffer b = new RollingAudioBuffer(4);
        b.add(new float[]{1, 2, 3, 4});
        b.add(new float[]{5, 6});
        assertArrayEquals(new float[]{3, 4, 5, 6}, b.lastWindow(4));
        assertEquals(6, b.totalWritten());
    }

    @Test
    void wrapsCorrectlyUnderManyAppends() {
        RollingAudioBuffer b = new RollingAudioBuffer(4);
        b.add(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        assertArrayEquals(new float[]{7, 8, 9, 10}, b.lastWindow(4));
    }
}
