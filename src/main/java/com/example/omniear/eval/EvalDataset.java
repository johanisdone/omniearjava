package com.example.omniear.eval;

import com.example.omniear.detection.ThreatCategory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Loads {@code labels.csv} for an evaluation dataset. */
public final class EvalDataset {

    private EvalDataset() {
    }

    /** Format: {@code file,label,onsetMs} with header; onsetMs may be blank. */
    public static List<EvalCase> load(Path csv) throws Exception {
        Path base = csv.toAbsolutePath().getParent();
        List<EvalCase> cases = new ArrayList<>();
        List<String> lines = Files.readAllLines(csv);
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split(",");
            if (i == 0 && parts[0].equalsIgnoreCase("file")) {
                continue; // header
            }
            if (parts.length < 2) {
                throw new IllegalArgumentException("bad row in " + csv + ": " + line);
            }
            Path file = base.resolve(parts[0].trim()).normalize();
            ThreatCategory label = parseLabel(parts[1].trim());
            long onset = parts.length >= 3 && !parts[2].isBlank() ? Long.parseLong(parts[2].trim()) : -1;
            cases.add(new EvalCase(file, label, onset));
        }
        if (cases.isEmpty()) {
            throw new IllegalArgumentException("no cases found in " + csv);
        }
        return cases;
    }

    /**
     * Maps a dataset label to a {@link ThreatCategory}. The legacy labels
     * {@code GUNSHOT} and {@code EXPLOSION} are folded into the merged
     * {@link ThreatCategory#GUNSHOT_EXPLOSION} category.
     */
    private static ThreatCategory parseLabel(String raw) {
        String name = raw.toUpperCase();
        if (name.equals("GUNSHOT") || name.equals("EXPLOSION")) {
            return ThreatCategory.GUNSHOT_EXPLOSION;
        }
        return ThreatCategory.valueOf(name);
    }
}
