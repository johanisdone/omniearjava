package com.example.omniear.audio;

import java.io.IOException;
import java.nio.file.Path;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * File-backed audio source used to replay WAV fixtures through the same
 * streaming pipeline as the microphone (deterministic, no hardware needed).
 */
public final class FileAudioSource implements AudioSource {

    private final AudioInputStream stream;
    private final AudioFormat format;

    public FileAudioSource(Path file) throws IOException, UnsupportedAudioFileException {
        AudioInputStream raw = AudioSystem.getAudioInputStream(file.toFile());
        AudioFormat src = raw.getFormat();
        // Normalize to PCM_SIGNED 16-bit so the adapter can assume 2 bytes/sample.
        if (src.getEncoding().equals(AudioFormat.Encoding.PCM_SIGNED)
                && src.getSampleSizeInBits() == 16) {
            this.stream = raw;
            this.format = src;
        } else {
            AudioFormat pcm16 = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    src.getSampleRate(), 16, src.getChannels(),
                    src.getChannels() * 2, src.getSampleRate(), false);
            this.stream = AudioSystem.getAudioInputStream(pcm16, raw);
            this.format = pcm16;
        }
    }

    @Override
    public AudioFormat format() {
        return format;
    }

    @Override
    public int read(byte[] buffer, int offset, int len) throws IOException {
        return stream.read(buffer, offset, len);
    }

    @Override
    public void close() {
        try {
            stream.close();
        } catch (IOException ignored) {
            // ignore
        }
    }
}
