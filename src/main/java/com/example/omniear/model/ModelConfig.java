package com.example.omniear.model;

import com.google.gson.Gson;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Metadata for the AudioDet ONNX head (class order, thresholds, windowing),
 * parsed from {@code onnx/audiodet_v4.json}.
 *
 * <p>This is the metadata for the ONNX-consistent head {@code onnx/head.onnx}.
 * It must NOT be replaced by {@code heads/*&#47;head_meta.json}, which describes
 * the ced.cpp (GGUF) head and is numerically inconsistent with the ONNX backbone.
 */
public final class ModelConfig {

    public List<String> classes;
    public Map<String, Double> thresholds;
    public int sr = 16000;
    public double win_s = 2.0;
    public double hop_s = 1.0;

    private static final Gson GSON = new Gson();

    public static ModelConfig load(Path json) throws IOException {
        try (Reader r = Files.newBufferedReader(json, StandardCharsets.UTF_8)) {
            return validate(GSON.fromJson(r, ModelConfig.class));
        }
    }

    public static ModelConfig loadDefault() throws IOException {
        try (InputStream in = ModelConfig.class.getResourceAsStream("/config/audiodet_v4.json")) {
            if (in == null) {
                throw new IOException("missing classpath resource /config/audiodet_v4.json");
            }
            ModelConfig cfg = GSON.fromJson(new InputStreamReader(in, StandardCharsets.UTF_8), ModelConfig.class);
            return validate(cfg);
        }
    }

    private static ModelConfig validate(ModelConfig cfg) {
        if (cfg == null || cfg.classes == null || cfg.classes.isEmpty()) {
            throw new IllegalStateException("model config has no classes");
        }
        if (cfg.thresholds == null) {
            throw new IllegalStateException("model config has no thresholds");
        }
        return cfg;
    }

    public int windowSamples() {
        return (int) Math.round(win_s * sr);
    }

    public int hopSamples() {
        return (int) Math.round(hop_s * sr);
    }

    public float threshold(String label) {
        Double v = thresholds.get(label);
        return v == null ? 0.5f : v.floatValue();
    }
}
