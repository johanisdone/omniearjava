package com.example.omniear.audio;

import java.util.Arrays;

/** A timestamped window of mono float32 samples at the model sample rate. */
public record AudioWindow(float[] samples, long startSample, long timestampMs, long capturedWallMs) {

    public int length() {
        return samples.length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof AudioWindow other)) {
            return false;
        }
        return startSample == other.startSample
                && timestampMs == other.timestampMs
                && capturedWallMs == other.capturedWallMs
                && Arrays.equals(samples, other.samples);
    }

    @Override
    public int hashCode() {
        int h = Arrays.hashCode(samples);
        h = 31 * h + Long.hashCode(startSample);
        h = 31 * h + Long.hashCode(timestampMs);
        h = 31 * h + Long.hashCode(capturedWallMs);
        return h;
    }

    @Override
    public String toString() {
        return "AudioWindow[start=" + startSample + ", ts=" + timestampMs
                + "ms, len=" + samples.length + "]";
    }
}
