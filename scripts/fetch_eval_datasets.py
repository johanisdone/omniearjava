"""Download public labelled audio for OmniEar real-data evaluation.

Sources:
  nigens  NIGENS general sound events db (Zenodo 2535878): screams, crash,
          speech/dog/baby/knock negatives. Fetched via HTTP Range on the remote
          zip so only the needed classes are downloaded.
  sesa    Sound Events for Surveillance Applications (Zenodo 3519845, 26 MB):
          gunshot, explosion, siren, casual.
  esc50   ESC-50 (github.com/karoldvl/ESC-50, direct raw fetch): glass_breaking,
          siren, fireworks, plus confusable negatives.

All audio is normalised to mono 16-bit PCM WAV so the Java harness can decode it.
Output layout:
  <out>/audio/<source>_<name>.wav
  <out>/manifests/<source>.csv
  <out>/labels.csv              (merge of all manifests; harness input)

Downloads are cached under .datasets-cache/ and each source is skipped if its
manifest already exists (use --force to refetch).

Usage (reference interpreter provides soundfile):
  python scripts/fetch_eval_datasets.py --out eval-real
"""

import argparse
import http.client
import io
import shutil
import struct
import subprocess
import sys
import zlib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NIGENS_URL = "https://zenodo.org/api/records/2535878/files/NIGENS.zip/content"
NIGENS_SIZE = 2174918847
SESA_URL = "https://zenodo.org/api/records/3519845/files/SESA.zip/content"
ESC50_META = "/karoldvl/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO = "/karoldvl/ESC-50/master/audio/{}"

SIG_EOCD = 0x06054b50
SIG_CDFH = 0x02014b50
SIG_LFH = 0x04034b50
SIG_Z64_LOC = 0x07064b50

NIGENS_POS = {"femaleScream": "SCREAM_DISTRESS", "maleScream": "SCREAM_DISTRESS",
              "crash": "IMPACT_CRASH"}
NIGENS_NEG = {"femaleSpeech": "NORMAL", "maleSpeech": "NORMAL", "dog": "NORMAL",
              "knock": "NORMAL", "baby": "NORMAL"}

SESA_MAP = {"gunshot": "GUNSHOT", "explosion": "EXPLOSION", "siren": "SIREN",
            "casual": "NORMAL"}

ESC50_MAP = {
    "glass_breaking": "GLASS_BREAK",
    "siren": "SIREN",
    "fireworks": "EXPLOSION",
    "car_horn": "NORMAL",
    "crying_baby": "NORMAL",
    "dog": "NORMAL",
    "clapping": "NORMAL",
    "chainsaw": "NORMAL",
    "clock_alarm": "NORMAL",
    "vacuum_cleaner": "NORMAL",
    "door_wood_knock": "NORMAL",
}


def _curl():
    return shutil.which("curl.exe") or shutil.which("curl")


def http_range(url, start, end, retries=4):
    args = [_curl(), "-sL", "--fail", "-r", f"{start}-{end}", url]
    for attempt in range(retries):
        try:
            out = subprocess.run(args, capture_output=True, timeout=900).stdout
            want = end - start + 1
            if len(out) == want:
                return out
            raise IOError(f"range {start}-{end}: got {len(out)} of {want}")
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}: {exc}", file=sys.stderr)
    raise RuntimeError("unreachable")


def http_download(url, dest, retries=4):
    for attempt in range(retries):
        rc = subprocess.run([_curl(), "-sL", "--fail", url, "-o", str(dest)],
                            timeout=3600).returncode
        if rc == 0 and dest.exists() and dest.stat().st_size > 0:
            return
        print(f"    retry download {attempt + 1}", file=sys.stderr)
    raise RuntimeError(f"download failed: {url}")


class HttpSession:
    def __init__(self, host):
        self.host = host
        self.conn = http.client.HTTPSConnection(host, timeout=180)

    def get(self, path, retries=4):
        for attempt in range(retries):
            try:
                self.conn.request("GET", path, headers={"User-Agent": "omniear-fetch"})
                resp = self.conn.getresponse()
                data = resp.read()
                if resp.status == 200:
                    return data
                raise IOError(f"HTTP {resp.status} for {path}")
            except Exception:  # noqa: BLE001
                self.conn.close()
                self.conn = http.client.HTTPSConnection(self.host, timeout=180)
                if attempt == retries - 1:
                    raise
        raise RuntimeError("unreachable")


def convert_to_pcm16(raw_bytes, dest):
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), mono, sr, subtype="PCM_16")
    return sr, len(mono) / sr


def parse_central_directory(url, size):
    tail = http_range(url, max(0, size - 65557), size - 1)
    eocd = tail.rfind(struct.pack("<I", SIG_EOCD))
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    _, _, _, _, total, cdsize, cdoff = struct.unpack_from("<IHHHHII", tail, eocd)
    if total == 0xFFFF or cdsize == 0xFFFFFFFF or cdoff == 0xFFFFFFFF:
        loc = tail.rfind(struct.pack("<I", SIG_Z64_LOC))
        (z64off,) = struct.unpack_from("<Q", tail, loc + 8)
        rec = http_range(url, z64off, z64off + 55)
        _, _, _, _, _, _, total, cdsize, cdoff = struct.unpack_from("<IQHHIIQQQQ", rec, 0)
    cd = http_range(url, cdoff, cdoff + cdsize - 1)
    entries = []
    p = 0
    while p + 46 <= len(cd) and struct.unpack_from("<I", cd, p)[0] == SIG_CDFH:
        (_, _, _, _, method, _, _, crc, csize, usize,
         nlen, elen, clen, _, _, _, lho) = struct.unpack_from("<IHHHHHHIIIHHHHHII", cd, p)
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            ep = 0
            while ep + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, ep)
                if hid == 0x0001:
                    vp = ep + 4
                    if usize == 0xFFFFFFFF:
                        (usize,) = struct.unpack_from("<Q", extra, vp); vp += 8
                    if csize == 0xFFFFFFFF:
                        (csize,) = struct.unpack_from("<Q", extra, vp); vp += 8
                    if lho == 0xFFFFFFFF:
                        (lho,) = struct.unpack_from("<Q", extra, vp); vp += 8
                ep += 4 + hsz
        entries.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "lho": lho})
        p += 46 + nlen + elen + clen
    return entries


def nigens_class(name):
    parts = name.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "NIGENS" else None


def nigens_candidate(entry):
    n = entry["name"].lower()
    return n.endswith(".wav") and not n.endswith(".wm.wav") \
        and nigens_class(entry["name"]) is not None


def extract_region(url, rows):
    start = min(e["lho"] for e in rows)
    end = max(e["lho"] + e["csize"] + 1024 for e in rows)
    blob = http_range(url, start, end)
    out = []
    for e in rows:
        rel = e["lho"] - start
        if struct.unpack_from("<I", blob, rel)[0] != SIG_LFH:
            out.append((e, http_range(url, e["lho"], e["lho"] + e["csize"] + 4096)))
            continue
        nlen, elen = struct.unpack_from("<HH", blob, rel + 26)
        data_off = rel + 30 + nlen + elen
        comp = blob[data_off:data_off + e["csize"]]
        if e["method"] == 8:
            raw = zlib.decompress(comp, -15)
        elif e["method"] == 0:
            raw = comp
        else:
            raise RuntimeError(f"unsupported method {e['method']}")
        out.append((e, raw))
    return out


def fetch_nigens(out, pos_per_class, neg_per_class, include_neg):
    print("[nigens] reading central directory ...", flush=True)
    entries = parse_central_directory(NIGENS_URL, NIGENS_SIZE)
    pos = {}
    for cls in NIGENS_POS:
        rows = sorted((e for e in entries if nigens_candidate(e) and nigens_class(e["name"]) == cls),
                      key=lambda e: e["lho"])[:pos_per_class]
        pos[cls] = rows
    neg = {}
    if include_neg:
        for cls in NIGENS_NEG:
            rows = sorted((e for e in entries if nigens_candidate(e) and nigens_class(e["name"]) == cls),
                          key=lambda e: e["lho"])[:neg_per_class]
            neg[cls] = rows
    rows = []
    plan = [(c, NIGENS_POS[c], pos[c]) for c in pos] + \
           [(c, NIGENS_NEG[c], neg.get(c, [])) for c in neg]
    for cls, cat, chosen in plan:
        if not chosen:
            continue
        print(f"[nigens] {cls}: {len(chosen)} clips -> {cat}", flush=True)
        for e, raw in extract_region(NIGENS_URL, chosen):
            base = Path(e["name"].stem).name
            dest = out / "audio" / f"nigens_{cls}_{base}.wav"
            try:
                convert_to_pcm16(raw, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"    skip {e['name']}: {exc}", file=sys.stderr)
                continue
            rows.append((f"audio/{dest.name}", cat))
    return rows


def fetch_sesa(out, cache, per_class):
    zpath = cache / "SESA.zip"
    if not zpath.exists():
        print("[sesa] downloading 26 MB ...", flush=True)
        http_download(SESA_URL, zpath)
    print("[sesa] extracting ...", flush=True)
    z = zipfile.ZipFile(zpath)
    counts = {}
    rows = []
    for name in sorted(z.namelist()):
        low = name.lower()
        if not low.endswith(".wav"):
            continue
        prefix = Path(name).name.split("_")[0].lower()
        cat = SESA_MAP.get(prefix)
        if cat is None:
            continue
        counts[prefix] = counts.get(prefix, 0) + 1
        if per_class and counts[prefix] > per_class:
            continue
        raw = z.read(name)
        parts = name.split("/")
        split = parts[-2] if len(parts) >= 2 else "x"
        dest = out / "audio" / f"sesa_{split}_{Path(name).name}"
        try:
            convert_to_pcm16(raw, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"    skip {name}: {exc}", file=sys.stderr)
            continue
        rows.append((f"audio/{dest.name}", cat))
    for k in sorted(counts):
        print(f"[sesa] {k}: {min(counts[k], per_class or counts[k])} clips -> {SESA_MAP[k]}", flush=True)
    return rows


def fetch_esc50(out, cache, per_class):
    meta = cache / "esc50.csv"
    if not meta.exists():
        print("[esc50] downloading metadata ...", flush=True)
        http_download("https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv", meta)
    picks = {k: [] for k in ESC50_MAP}
    for line in meta.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        fname, _fold, _target, cat = line.split(",")[:4]
        if cat in picks and len(picks[cat]) < per_class:
            picks[cat].append(fname)
    session = HttpSession("raw.githubusercontent.com")
    cache_audio = cache / "esc50" / "audio"
    cache_audio.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat, fnames in picks.items():
        if not fnames:
            continue
        print(f"[esc50] {cat}: {len(fnames)} clips -> {ESC50_MAP[cat]}", flush=True)
        for fname in fnames:
            cached = cache_audio / fname
            if cached.exists():
                raw = cached.read_bytes()
            else:
                raw = session.get(ESC50_AUDIO.format(fname))
                cached.write_bytes(raw)
            dest = out / "audio" / f"esc50_{fname}"
            try:
                convert_to_pcm16(raw, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"    skip {fname}: {exc}", file=sys.stderr)
                continue
            rows.append((f"audio/{dest.name}", ESC50_MAP[cat]))
    return rows


SOURCES = ("nigens", "sesa", "esc50")


def manifest_path(out, source):
    return out / "manifests" / f"{source}.csv"


def load_manifest(out, source):
    p = manifest_path(out, source)
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel, cat = line.split(",")[:2]
            rows.append((rel, cat))
    return rows


def save_manifest(out, source, rows):
    p = manifest_path(out, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"{r},{c}" for r, c in rows) + "\n", encoding="utf-8")


def bootstrap_from_labels(out):
    labels = out / "labels.csv"
    if not labels.exists():
        return
    by = {s: [] for s in SOURCES}
    for line in labels.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lower().startswith("file,"):
            continue
        rel, cat = line.split(",")[:2]
        for s in SOURCES:
            if Path(rel).name.startswith(s + "_"):
                by[s].append((rel, cat))
    for s in SOURCES:
        if by[s] and not manifest_path(out, s).exists():
            save_manifest(out, s, by[s])
            print(f"[{s}] imported {len(by[s])} clips from existing labels.csv", flush=True)


def merge_labels(out):
    lines = ["file,label,onsetMs"]
    total = 0
    for s in SOURCES:
        rows = load_manifest(out, s) or []
        for rel, cat in sorted(rows):
            lines.append(f"{rel},{cat},")
            total += 1
    (out / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval-real", type=Path)
    ap.add_argument("--cache", default=".datasets-cache", type=Path)
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--force", action="store_true", help="refetch even if manifest exists")
    ap.add_argument("--pos-per-class", type=int, default=40, help="NIGENS positives/class")
    ap.add_argument("--neg-per-class", type=int, default=15, help="NIGENS negatives/class")
    ap.add_argument("--no-negatives", action="store_true", help="skip NIGENS negatives")
    ap.add_argument("--sesa-per-class", type=int, default=0, help="0 = all SESA clips")
    ap.add_argument("--esc50-per-class", type=int, default=20, help="ESC-50 clips/class")
    args = ap.parse_args()

    out = args.out.resolve()
    cache = (ROOT / args.cache if not args.cache.is_absolute() else args.cache).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    bootstrap_from_labels(out)

    for source in args.datasets:
        if load_manifest(out, source) is not None and not args.force:
            print(f"[{source}] already present, skipping", flush=True)
            continue
        if source == "nigens":
            rows = fetch_nigens(out, args.pos_per_class, args.neg_per_class, not args.no_negatives)
        elif source == "sesa":
            rows = fetch_sesa(out, cache, args.sesa_per_class)
        elif source == "esc50":
            rows = fetch_esc50(out, cache, args.esc50_per_class)
        else:
            continue
        save_manifest(out, source, rows)
        pos = sum(1 for r in rows if r[1] != "NORMAL")
        print(f"[{source}] wrote {len(rows)} clips ({pos} positive, {len(rows) - pos} negative)\n",
              flush=True)

    total = merge_labels(out)
    print(f"merged {total} clips -> {out / 'labels.csv'}", flush=True)


if __name__ == "__main__":
    main()
