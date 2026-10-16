package com.example.omniear.ui;

import com.example.omniear.detection.ThreatCategory;
import java.awt.Color;
import java.util.List;

/**
 * Local alert priorities, declared worst first so {@link #ordinal()} increases
 * with mildness. Each priority carries the colour and blink rate of the visual
 * indicator and the synthesized tone played once per alert.
 */
enum AlertPriority {

    /** Critical: gunshots and explosions. Fast red blink, urgent two-tone alarm. */
    P0("P0", new Color(0xff5c5c), 220, List.of(
            new Tone(880, 110, 0.60f), Tone.silence(50),
            new Tone(440, 110, 0.60f), Tone.silence(50),
            new Tone(880, 110, 0.60f), Tone.silence(50),
            new Tone(440, 170, 0.60f))),

    /** High: scream/distress, impact/crash and glass break. Amber double beep. */
    P1("P1", new Color(0xffb020), 420, List.of(
            new Tone(880, 130, 0.42f), Tone.silence(70), new Tone(880, 130, 0.42f))),

    /** Informational: sirens. Slow blue blink, single soft beep. */
    P2("P2", new Color(0x5b8cff), 700, List.of(
            new Tone(660, 180, 0.25f)));

    /** One tone segment: {@code hz == 0} is silence, {@code amp} is 0..1. */
    record Tone(double hz, int ms, float amp) {
        static Tone silence(int ms) {
            return new Tone(0, ms, 0f);
        }
    }

    private final String label;
    private final Color color;
    private final int blinkMs;
    private final List<Tone> tones;

    AlertPriority(String label, Color color, int blinkMs, List<Tone> tones) {
        this.label = label;
        this.color = color;
        this.blinkMs = blinkMs;
        this.tones = tones;
    }

    String label() {
        return label;
    }

    Color color() {
        return color;
    }

    /** Milliseconds for one blink on/off phase. */
    int blinkMs() {
        return blinkMs;
    }

    List<Tone> tones() {
        return tones;
    }

    /** True when this priority is at least as severe as {@code other}. */
    boolean atLeast(AlertPriority other) {
        return ordinal() <= other.ordinal();
    }

    /** Priority for a threat category, or {@code null} for non-alerting ones. */
    static AlertPriority of(ThreatCategory category) {
        return switch (category) {
            case GUNSHOT_EXPLOSION -> P0;
            case SCREAM_DISTRESS, IMPACT_CRASH, GLASS_BREAK -> P1;
            case SIREN -> P2;
            default -> null;
        };
    }
}
