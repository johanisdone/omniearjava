package com.example.omniear.model;

import java.util.List;

/**
 * Small interface around threat-model inference.
 *
 * <p>A prediction consumes exactly one model window of mono float32 PCM in
 * [-1, 1] at the model sample rate (see {@link ModelConfig}).
 */
public interface ThreatModel extends AutoCloseable {

    ModelPrediction predict(float[] window);

    List<String> labels();

    @Override
    void close();
}
