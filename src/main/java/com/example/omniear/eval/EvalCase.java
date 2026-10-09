package com.example.omniear.eval;

import com.example.omniear.detection.ThreatCategory;
import java.nio.file.Path;

/**
 * One labelled evaluation clip.
 *
 * @param file    audio file
 * @param label   expected category ({@link ThreatCategory#NORMAL} for background/negative)
 * @param onsetMs expected event onset in audio time, or -1 if unknown
 */
public record EvalCase(Path file, ThreatCategory label, long onsetMs) {
}
