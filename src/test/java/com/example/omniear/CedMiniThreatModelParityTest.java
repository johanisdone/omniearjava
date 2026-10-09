package com.example.omniear;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.example.omniear.model.AudioPreprocessor;
import com.example.omniear.model.CedMiniThreatModel;
import com.example.omniear.model.ModelConfig;
import com.example.omniear.model.ModelPrediction;
import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Phase 2 exit criterion: Java inference must reproduce the Python reference
 * scores for the same WAV within a small numerical tolerance.
 */
class CedMiniThreatModelParityTest {

    private static final Path MODELS = Path.of("models");
    private static final Path AUDIO = Path.of("src", "test", "resources", "audio");
    private static final float TOLERANCE = 1e-3f;

    static class Golden {
        @SerializedName("window_offsets")
        List<Integer> windowOffsets;
        @SerializedName("window_probabilities")
        List<List<Double>> windowProbabilities;
        List<String> classes;
        List<String> fired;
    }

    static Stream<Path> fixtures() throws Exception {
        return Files.list(AUDIO)
                .filter(p -> p.toString().endsWith(".wav"))
                .sorted();
    }

    @ParameterizedTest
    @MethodSource("fixtures")
    void matchesPythonReference(Path wav) throws Exception {
        Path goldenPath = AUDIO.resolve(wav.getFileName().toString().replace(".wav", ".json"));
        assertTrue(Files.isRegularFile(goldenPath), "missing golden file " + goldenPath);
        Golden golden = new Gson().fromJson(Files.readString(goldenPath), Golden.class);
        assertNotNull(golden.windowProbabilities);

        ModelConfig config = ModelConfig.loadDefault();
        float[] samples = AudioPreprocessor.readMono(wav, config.sr);
        int[] offsets = AudioPreprocessor.windowOffsets(
                samples.length, config.windowSamples(), config.hopSamples());

        assertEquals(golden.windowOffsets.size(), offsets.length, "window count mismatch");
        for (int i = 0; i < offsets.length; i++) {
            assertEquals(golden.windowOffsets.get(i).intValue(), offsets[i], "window offset " + i);
        }

        try (CedMiniThreatModel model = CedMiniThreatModel.fromModelsDir(MODELS, config)) {
            float[][] windows = AudioPreprocessor.windows(samples, config);
            float[][] probs = model.predictBatch(windows);
            assertEquals(golden.windowProbabilities.size(), probs.length);
            for (int w = 0; w < probs.length; w++) {
                List<Double> expected = golden.windowProbabilities.get(w);
                for (int c = 0; c < probs[w].length; c++) {
                    float exp = expected.get(c).floatValue();
                    float act = probs[w][c];
                    assertEquals(exp, act, TOLERANCE,
                            wav.getFileName() + " window " + w + " class "
                                    + golden.classes.get(c) + " expected=" + exp + " actual=" + act);
                }
            }
        }
    }

    @Test
    void labelsMatchGoldenOrder() throws Exception {
        Golden golden = new Gson().fromJson(
                Files.readString(AUDIO.resolve("silence_2s.json")), Golden.class);
        ModelConfig config = ModelConfig.loadDefault();
        assertEquals(golden.classes, config.classes, "class order must match reference");
    }
}
