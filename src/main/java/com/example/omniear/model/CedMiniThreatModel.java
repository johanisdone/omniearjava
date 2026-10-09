package com.example.omniear.model;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Adapter for the AudioDet CED-mini checkpoint via ONNX Runtime.
 *
 * <p>Two-stage ONNX graph, both stages run through ONNX Runtime:
 * <pre>
 *   ced-mini.onnx : waveform[1,32000] -> logits[1,527]
 *   head.onnx     : logits[1,527]     -> probabilities[1,17]
 * </pre>
 * Both sessions are created once and reused; {@link OrtSession#run} is
 * thread-safe, so a single instance can serve the capture worker.
 */
public final class CedMiniThreatModel implements ThreatModel {

    private final OrtEnvironment env;
    private final OrtSession backbone;
    private final OrtSession head;
    private final String backboneInput;
    private final String headInput;
    private final ModelConfig config;
    private final int windowSamples;

    public static CedMiniThreatModel fromModelsDir(Path modelsDir, ModelConfig config)
            throws OrtException {
        return new CedMiniThreatModel(
                OrtEnvironment.getEnvironment(),
                modelsDir.resolve("ced-mini.onnx"),
                modelsDir.resolve("onnx").resolve("head.onnx"),
                config);
    }

    public CedMiniThreatModel(Path backboneOnnx, Path headOnnx, ModelConfig config)
            throws OrtException {
        this(OrtEnvironment.getEnvironment(), backboneOnnx, headOnnx, config);
    }

    public CedMiniThreatModel(OrtEnvironment env, Path backboneOnnx, Path headOnnx, ModelConfig config)
            throws OrtException {
        this.env = env;
        this.config = config;
        this.windowSamples = config.windowSamples();
        require(backboneOnnx);
        require(headOnnx);
        this.backbone = env.createSession(backboneOnnx.toString(), new OrtSession.SessionOptions());
        this.head = env.createSession(headOnnx.toString(), new OrtSession.SessionOptions());
        this.backboneInput = backbone.getInputNames().iterator().next();
        this.headInput = head.getInputNames().iterator().next();
    }

    private static void require(Path p) {
        if (!Files.isRegularFile(p)) {
            throw new IllegalStateException("missing model file: " + p.toAbsolutePath());
        }
    }

    @Override
    public ModelPrediction predict(float[] window) {
        if (window.length != windowSamples) {
            throw new IllegalArgumentException(
                    "expected " + windowSamples + " samples, got " + window.length);
        }
        float[] probs = predictBatch(new float[][]{window})[0];
        return new ModelPrediction(config.classes, probs);
    }

    /** Run a batch of windows (row-major [n][windowSamples]) through both stages. */
    public float[][] predictBatch(float[][] windows) {
        for (float[] w : windows) {
            if (w.length != windowSamples) {
                throw new IllegalArgumentException(
                        "expected " + windowSamples + " samples, got " + w.length);
            }
        }
        int n = windows.length;
        float[] flat = new float[n * windowSamples];
        for (int i = 0; i < n; i++) {
            System.arraycopy(windows[i], 0, flat, i * windowSamples, windowSamples);
        }
        try (OnnxTensor wf = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{n, windowSamples});
             OrtSession.Result res = backbone.run(Map.of(backboneInput, wf))) {
            float[][] logits = (float[][]) res.get(0).getValue();
            return runHead(logits);
        } catch (OrtException e) {
            throw new IllegalStateException("backbone inference failed", e);
        }
    }

    private float[][] runHead(float[][] logits) throws OrtException {
        if (logits.length == 1) {
            float[] row = logits[0];
            try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(row), new long[]{1, row.length});
                 OrtSession.Result res = head.run(Map.of(headInput, t))) {
                float[][] probs = (float[][]) res.get(0).getValue();
                return new float[][]{probs[0].clone()};
            }
        }
        int dim = logits[0].length;
        float[] flat = new float[logits.length * dim];
        for (int i = 0; i < logits.length; i++) {
            System.arraycopy(logits[i], 0, flat, i * dim, dim);
        }
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), new long[]{logits.length, dim});
             OrtSession.Result res = head.run(Map.of(headInput, t))) {
            float[][] probs = (float[][]) res.get(0).getValue();
            float[][] out = new float[probs.length][];
            for (int i = 0; i < probs.length; i++) {
                out[i] = probs[i].clone();
            }
            return out;
        }
    }

    @Override
    public List<String> labels() {
        return config.classes;
    }

    public ModelConfig config() {
        return config;
    }

    @Override
    public void close() {
        try {
            head.close();
            backbone.close();
        } catch (OrtException e) {
            throw new IllegalStateException("failed to close ONNX sessions", e);
        }
    }
}
