# Golden fixtures of the eodata readers

What the data pipeline of deep_lst returned while it still read the `.eozarr` stores through
eodata (v0.3_DeepLST, zarr 3.2): ten training anchors, four windows of four days, four
modalities. The readers of `deep_lst/stores/` and the normaliser are tested against these files
(`tests/test_stores_parity.py`, `tests/test_stores_grid.py`, `tests/test_stores_anchors.py`,
`tests/test_normalisation.py`).

| File | Content |
|---|---|
| `<modality>.npz` | anchor ids and coordinates, store rows and cell coordinates, and per leaf the values as read (`raw__<leaf>`) and after normalisation (`normalized__<leaf>`) |
| `meta.json` | pixels, windows, variable names of every leaf, stores, versions and commits |
| `statistics.json` | the statistics (mean, std, min, max) that normalised the values |

Temporal leaves have the shape (windows, pixels, 96 hours, variables); position h of window w is
the hour `start_days[w] * 24 + h` since 1970-01-01T00. Static leaves have the shape
(pixels, variables).

The files were written by `scripts/assets/export_reader_fixtures.py`, which needed the eodata
readers and was removed together with them. It is in the history of the repository; its last
version is in commit 63afdcf (`git show 63afdcf:scripts/assets/export_reader_fixtures.py`).

These files cannot be regenerated from the current code and must not be: they are the only
record of what the old pipeline returned.
