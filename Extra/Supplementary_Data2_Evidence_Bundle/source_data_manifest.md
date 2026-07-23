# Source-Data Manifest

## Original CICIDS2017 source captures

The 1771-event corpus used throughout this study (`locked_dataset.csv`) was built by
`Prepare_dataset.ipynb` (vendored in this bundle at `vendored_repo/Extra/`, see below)
from exactly three named CICIDS2017 capture files, published by the Canadian Institute
for Cybersecurity, University of New Brunswick (https://www.unb.ca/cic/datasets/ids-2017.html):

1. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
2. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
3. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

## Row selection and label mapping

From these three files, a fixed selection plan retained: 750 BENIGN, 250 DDoS, 250
PortScan, 250 Web Attack -- Brute Force, 250 Web Attack -- XSS, and 21 Web Attack --
SQL Injection records (total 1771). Multi-class labels are mapped to a binary
`target_binary` column: `BENIGN` -> 0, every attack label -> 1 (533/778/etc. split
across partitions -- see `split_summary.json`).

## Retained vs. excluded fields

- **Retained (model-facing, 77 columns)**: the standard CICIDS2017 flow-statistics
  columns (duration, packet counts/lengths, IAT statistics, flag counts, window
  sizes, active/idle statistics, etc.) -- see the header row of
  `cicids2017_leakage_safe_without_target_binary.csv` for the exact ordered list.
- **Excluded from the model-facing representation**: `Source Port`, `Destination
  Port`, and `Protocol` (`EXCLUDE_PORTS_FROM_MODEL=True` in the prep notebook), to
  avoid IP/port shortcut learning. These fields WERE used to build the *original*
  fingerprint for `Prepare_dataset.ipynb`'s initial group-disjoint split -- which is
  exactly why that original split diverged from a fingerprint computed on the
  reduced, port-stripped feature set (the 9.88% "overlap" discussed in the main
  manuscript, Section "Locked Dataset and Split Protocol").
- **Row identifier**: `row_id` (format `cicids1771::<index>`), immutable, carried
  through every downstream file in this bundle (`split_manifest.csv`,
  `locked_dataset.csv`, all `predictions/*.npz`).

## Files and checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `cicids2017_leakage_safe.csv` (source, with target_binary and ports) | `60ddf7b2f9fafa9934bb3cdfdbb0abdc30cf080d81f01ab5ab53a179cefcd0a7` |
| `cicids2017_leakage_safe_without_target_binary.csv` (source, model-facing 77 cols + label/split/row_id) | `7667397478549a0d25a72777a81859ee3f3ca526013d1539dcdc0cf21e504db6` |

**Redistribution status of these two files differs and is stated precisely here**
(an earlier draft of this manifest described both as withheld, which an external
audit correctly flagged as inconsistent with what the bundle actually contains):

- `cicids2017_leakage_safe_without_target_binary.csv` (the final, model-facing,
  port-stripped derived CSV) **is included** in this bundle, at
  `vendored_repo/attached_assets/cicids2017_leakage_safe_without_target_binary.csv`,
  matching the checksum above.
- `cicids2017_leakage_safe.csv` (the earlier intermediate CSV that still carries
  `Source Port`/`Destination Port`/`Protocol` and an unreduced `target_binary`
  column) is **not** redistributed in this bundle; it remains only in the original
  repository's `attached_assets/` folder.

`locked_dataset.csv` in this bundle is the exact 77-feature, row-id-keyed table
this study's scripts actually consume, regenerated deterministically from the
included without-target-binary CSV by `01_build_locked_split.py`.

## Raw-to-derived preparation notebook (for maximal provenance)

`vendored_repo/Extra/Prepare_dataset.ipynb` is included in this bundle. It documents,
cell by cell, exactly how the 1771-event corpus was selected and partitioned from the
raw CICIDS2017 capture files: the identifier columns excluded before feature
selection, the fixed per-class row-selection plan, the feature-fingerprint-based
group-holdout split, and the port-exclusion step for the model-facing representation.
It cannot be executed standalone in this bundle: it was authored for Google Colab
(paths under `/content/...`) and expects the raw CICIDS2017 capture files as a
Colab-uploaded zip archive, which is not redistributed here for size and licensing
reasons -- the raw captures must be obtained directly from the Canadian Institute for
Cybersecurity (link above). We include it for transformation-logic transparency, not
as an executable regeneration path; `01_build_locked_split.py`, operating on the
already-included without-target-binary CSV, is the actually-executable regeneration
path for everything downstream of it.

## Regeneration instructions

1. Obtain the three named CICIDS2017 capture files from the Canadian Institute for
   Cybersecurity (link above), or use the two checksummed intermediate CSVs above if
   already in hand.
2. Run `01_build_locked_split.py` (adjust the `SRC` path at the top of the script to
   point at `cicids2017_leakage_safe_without_target_binary.csv`) to regenerate
   `split_manifest.csv`, `split_summary.json`, and `locked_dataset.csv` from
   scratch, and to re-verify the zero-cross-partition-duplicate assertion.
3. Proceed with `02_train_models.py` onward as described in `README.md`.
