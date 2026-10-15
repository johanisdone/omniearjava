package com.example.omniear.audio;

/**
 * Bounded circular buffer of mono samples.
 *
 * <p>Keeps only the most recent {@code capacity} samples so a rolling window can
 * be produced without copying or retaining the full stream.
 */
public final class RollingAudioBuffer {

    private final float[] data;
    private final int capacity;
    private long totalWritten;

    public RollingAudioBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0");
        }
        this.capacity = capacity;
        this.data = new float[capacity];
    }

    public int capacity() {
        return capacity;
    }

    /** Total number of samples ever appended. */
    public synchronized long totalWritten() {
        return totalWritten;
    }

    public synchronized void add(float[] chunk) {
        for (float sample : chunk) {
            data[(int) (totalWritten % capacity)] = sample;
            totalWritten++;
        }
    }

    /**
     * Return the most recent {@code n} samples. Positions before the start of the
     * stream are zero-filled, so this is safe to call before {@code n} samples
     * have arrived.
     */
    public synchronized float[] lastWindow(int n) {
        return windowEndingAt(totalWritten, n);
    }

    /**
     * Return the {@code n} samples ending exactly at (exclusive) global index
     * {@code endExclusive}. Positions before the start of the stream are
     * zero-filled. The range must lie within the retained window
     * ({@code endExclusive - n >= totalWritten - capacity}).
     */
    public synchronized float[] windowEndingAt(long endExclusive, int n) {
        if (n > capacity) {
            throw new IllegalArgumentException("requested " + n + " > capacity " + capacity);
        }
        float[] out = new float[n];
        long start = endExclusive - n;
        for (int i = 0; i < n; i++) {
            long idx = start + i;
            out[i] = idx < 0 ? 0f : data[(int) Math.floorMod(idx, capacity)];
        }
        return out;
    }
}
