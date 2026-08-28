# benchmarks/

v0.4.0 scaffold for measuring the mujik-transcriptor pipeline.

## Scripts

### `run_separation.py`
Run Demucs (4-stem `htdemucs_ft` or 6-stem `htdemucs_6s`) and report timing + volume.

```bash
PYTHONPATH=src python benchmarks/run_separation.py \
    --input /path/to/song.wav \
    --variant htdemucs_6s \
    --device cuda \
    --out /tmp/separation_report.json
```

### `run_transcription.py`
Run a transcription adapter (basic-pitch / adtof / bytedance-piano) and report timing + note count.

```bash
PYTHONPATH=src python benchmarks/run_transcription.py \
    --input /path/to/song.wav \
    --adapter basic-pitch \
    --out /tmp/transcription_report.json
```

## v0.4.0 scope (scaffold only)

- These scripts record `elapsed_sec` + metadata (input path, adapter choice).
- They do **not** run heavy model inference in the scaffold mode (v0.4.0 only).
- Real evaluation (SDR/SIR/SAR for separation; onset F1 / pitch F1 for transcription) requires a labeled ground-truth dataset — out of scope for v0.4.0.

## Roadmap

- v0.5: bring in a small labeled corpus (e.g. MUSDB18 for separation, MAPS for piano, MIREX for onset) and wire `mir_eval` / `museval` to compute real metrics.
- v0.5+: 5-genre benchmark (pop / jazz / metal / electronic / folk) — requires a curated multi-genre dataset, deferred until dataset licensing is sorted out.
