package com.example.omniear.detection;

import com.google.gson.Gson;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Label &rarr; category mapping plus per-category decision rules, loaded from
 * {@code config/categories.json}. Keeping this in configuration avoids scattering
 * model label strings through the code.
 */
public final class DetectionConfig {

    /** Categories to evaluate, in priority order. */
    public List<CategoryRule> categories;

    /** Windows whose top category score is below this are considered "quiet". */
    public float uncertainFloor = 0.3f;

    private static final Gson GSON = new Gson();

    public static DetectionConfig load(Path json) throws IOException {
        try (Reader r = Files.newBufferedReader(json, StandardCharsets.UTF_8)) {
            return validate(GSON.fromJson(r, DetectionConfig.class));
        }
    }

    public static DetectionConfig loadDefault() throws IOException {
        try (InputStream in = DetectionConfig.class.getResourceAsStream("/config/categories.json")) {
            if (in == null) {
                throw new IOException("missing classpath resource /config/categories.json");
            }
            return validate(GSON.fromJson(
                    new InputStreamReader(in, StandardCharsets.UTF_8), DetectionConfig.class));
        }
    }

    private static DetectionConfig validate(DetectionConfig cfg) {
        if (cfg == null || cfg.categories == null || cfg.categories.isEmpty()) {
            throw new IllegalStateException("detection config has no categories");
        }
        return cfg;
    }
}
