# Separation Benchmark (MUSDB18 + museval)

- Variant: `bsroformer` (device=mps)
- Subset: test / 10 tracks
- **Mean SDR (per-stem median): 11.64 dB**
- Wall time: 6501.6s (sep 3554.1s + eval 2926.1s)

| Stem | SDR | SIR | SAR |
|---|---|---|---|
| vocals | 11.006 | 21.333 | 11.411 |
| drums | 12.998 | 23.773 | 13.649 |
| bass | 12.96 | 23.415 | 13.665 |
| other | 9.597 | 17.642 | 10.013 |

## Per-track

| Track | Duration | Sep time | vocals SDR | drums SDR | bass SDR | other SDR |
|---|---|---|---|---|---|---|
| The Easton Ellises (Baumi) - SDRNR | 234.7s | 281.1s | 10.863 | 13.114 | 10.497 | 7.454 |
| Bobby Nobody - Stitch Up | 221.2s | 309.1s | 13.619 | 10.467 | 12.734 | 10.245 |
| Al James - Schoolboy Facination | 200.5s | 280.1s | 11.808 | 9.547 | 13.186 | 7.323 |
| James Elder & Mark M Thompson - The English Actor | 205.5s | 281.1s | 11.15 | 12.782 | 10.178 | 10.053 |
| Girls Under Glass - We Feel Alright | 317.3s | 426.4s | 7.684 | 13.48 | 5.242 | 5.208 |
| Georgia Wonder - Siren | 430.4s | 592.5s | 8.744 | 14.402 | 15.768 | 9.663 |
| Buitraker - Revo X | 275.8s | 379.9s | 9.21 | 16.66 | 12.33 | 9.535 |
| Ben Carrigan - We'll Talk About It All Tonight | 254.9s | 372.6s | 9.927 | 12.044 | 13.95 | 9.796 |
| Side Effects Project - Sing With Me | 243.7s | 332.6s | 17.742 | 17.026 | 16.299 | 4.848 |
| BKS - Too Much | 220.1s | 298.7s | 15.072 | 12.881 | 15.744 | 9.659 |