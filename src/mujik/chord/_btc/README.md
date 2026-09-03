# Vendored: BTC-ISMIR19 (chord recognition)

Source: https://github.com/jayg996/BTC-ISMIR19 (MIT License, commit `master` @ 2020-05-23)

Paper: Jonggwon Park, Kyoyun Choi, Sungwook Jeon, Dokyun Kim & Jonghun Park,
"A Bi-Directional Transformer for Musical Chord Recognition", ISMIR 2019.

## Why vendored

`mujik.chord.btc_hcqt_adapter` runs BTC inference in a subprocess. The wrapper
imports this package's modules directly (`btc_model`, `utils.mir_eval_modules`,
`utils.hparams`) so users do not need to clone the upstream repo or set
`BTC_ISMIR19_PATH`.

## What is vendored (minimal import surface)

- `btc_model.py` — BTC model definition
- `utils/__init__.py`
- `utils/hparams.py` — HParams config loader
- `utils/mir_eval_modules.py` — feature extraction (`audio_file_to_features`),
  25-class `idx2chord`, 170-class `idx2voca_chord()`
- `utils/transformer_modules.py` — transformer layer building blocks
- `LICENSE` — upstream MIT license (kept verbatim)

The 170-class vocabulary (`idx2voca_chord()`) is generated in code — no
external data files needed. Note the label convention: bare root (`"C"`) =
**major**, `"C:min"` = minor.

NOT vendored: training code, pretrained weights (`.pt`). Weights are
downloaded at image build time (see `Dockerfile.ml`, ~12 MB
`btc_model_large_voca.pt`) and resolved at runtime via
`config.btc_model_path` → env `MUJIK_BTC_MODEL`.

## Local patches (keep when updating!)

- `utils/transformer_modules.py`: `np.float` → builtin `float` (removed in
  numpy 1.24+; same fix as madmom's, applied at vendor time).

## Updates

To update, re-copy the files above from upstream and keep this README +
upstream `LICENSE` in sync.
