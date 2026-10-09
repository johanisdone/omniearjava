package com.example.omniear.audio;

import java.io.IOException;
import javax.sound.sampled.AudioFormat;

/** A stream of raw PCM bytes (live microphone or file replay). */
public interface AudioSource extends AutoCloseable {

    AudioFormat format();

    /** Read up to {@code len} bytes; returns count read, or -1 at end of stream. */
    int read(byte[] buffer, int offset, int len) throws IOException;

    @Override
    void close();
}
