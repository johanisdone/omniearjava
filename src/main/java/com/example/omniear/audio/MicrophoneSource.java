package com.example.omniear.audio;

import java.io.IOException;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.TargetDataLine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Live microphone source backed by the Java Sound API.
 *
 * <p>Selects a mixer exposing a {@link TargetDataLine}: by name when
 * {@code deviceName} is given, otherwise the system default input. Any
 * OS-visible capture device (built-in, USB, Bluetooth) works.
 */
public final class MicrophoneSource implements AudioSource {

    private static final Logger LOG = LoggerFactory.getLogger(MicrophoneSource.class);

    private final TargetDataLine line;
    private final AudioFormat format;

    public MicrophoneSource(AudioFormat format, String deviceName) throws LineUnavailableException {
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
        this.line = deviceName == null
                ? (TargetDataLine) AudioSystem.getLine(info)
                : lineFromMixer(deviceName, info);
        this.format = line.getFormat();
        this.line.open(this.format);
        this.line.start();
        LOG.info("microphone open: {}", this.format);
    }

    private static TargetDataLine lineFromMixer(String deviceName, DataLine.Info info)
            throws LineUnavailableException {
        for (Mixer.Info mi : AudioSystem.getMixerInfo()) {
            if (!mi.getName().toLowerCase().contains(deviceName.toLowerCase())) {
                continue;
            }
            Mixer mixer = AudioSystem.getMixer(mi);
            if (mixer.isLineSupported(info)) {
                return (TargetDataLine) mixer.getLine(info);
            }
        }
        throw new LineUnavailableException("no capture device matching '" + deviceName + "'");
    }

    /** Names of mixers that can capture, for {@code --list-devices}. */
    public static String[] captureDeviceNames() {
        return AudioSystem.getMixerInfo().length == 0
                ? new String[0]
                : java.util.Arrays.stream(AudioSystem.getMixerInfo())
                        .filter(mi -> AudioSystem.getMixer(mi).getTargetLineInfo().length > 0)
                        .map(Mixer.Info::getName)
                        .toArray(String[]::new);
    }

    @Override
    public AudioFormat format() {
        return format;
    }

    @Override
    public int read(byte[] buffer, int offset, int len) throws IOException {
        return line.read(buffer, offset, len);
    }

    @Override
    public void close() {
        try {
            if (line.isOpen()) {
                line.stop();
                line.flush();
                line.close();
            }
        } catch (RuntimeException e) {
            LOG.warn("error closing microphone", e);
        }
    }
}
