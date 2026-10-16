package com.example.omniear.detection;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
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

    /**
     * Noise gate: windows whose RMS level is below this many dBFS are treated as
     * background (electrical hiss / room tone) and cannot raise an alert. Guards
     * against the model hallucinating events on near-silent input. Set very low
     * (e.g. -120) to disable.
     */
    public float noiseGateDb = -55f;

    private static final Gson GSON = new GsonBuilder()
            // config writes transform kinds in lower case ("movavg"); Gson's default
            // enum adapter is case-sensitive and would silently map it to null.
            .registerTypeAdapter(TransformKind.class, new JsonDeserializer<TransformKind>() {
                @Override
                public TransformKind deserialize(JsonElement json, java.lang.reflect.Type typeOfT,
                                                 JsonDeserializationContext context)
                        throws JsonParseException {
                    return TransformKind.parse(json.getAsString());
                }
            })
            .create();

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
