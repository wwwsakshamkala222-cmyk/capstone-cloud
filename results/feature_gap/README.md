# Feature gap: our CICIoT2023 release vs. Neto et al. (2023)

**Conclusion:** the v4 "paper's 46-feature protocol" run was not carried out. The 8 features
the paper uses that are missing from our cache do not exist in any CICIoT2023 copy on this machine,
so we cannot rebuild the paper's feature set. The 2/8/34-class comparison in the main README
still uses our 39 features, and that difference has to be stated next to any comparison with
the paper.

## Paper (46 features) vs. ours (39 features)

| | Features |
|---|---|
| In both (38) | Header_Length, Protocol Type, Rate, fin/syn/rst/psh/ack/ece/cwr flag numbers, ack/syn/fin/rst count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance, plus paper `Duration` = our `Time_To_Live` (\*) |
| Paper only (8) | **flow_duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight** |
| Ours only (1) | IGMP |

(\*) We treat these as the same field because the counts line up exactly: 46 − 8 missing = 38 shared,
and our 39 = 38 + IGMP. We did not check this against the paper's raw files, because we don't have them.

## What was checked (2026-11-20)

1. **Raw CSVs**: `/dev/shm/saksham/ciciot2023/CSV/` has 309 per-class `*.pcap.csv` files in 34 class
   folders. All 309 have the **same 39-column header**, which matches `src/preprocessing/data_meta.json`.
2. **Other copies on the machine**: searched by name (`*ciciot*`, `*merged*`, `*iot2023*`) across `/`,
   `/raid` and `/dev/shm`, limited to readable directories. The only hits were the copy above and an empty
   `~/saksham/data/ciciot2023/`. No `MERGED_CSV` folder exists.
3. **Search by header**: checked the first line of every readable `*.csv` under `/raid`, `/home`,
   `/dev/shm`, `/opt`, `/tmp`, `/srv` and `/mnt` for `flow_duration`, `Srate`, `Magnitue`, `Radius` or
   `Covariance`. **No file had any of them.**
4. **The dataset's own README** (`CSV/README_CSV.pdf`) describes two layouts: "169 .csv files … a
   combined and shuffled dataset including all attacks … identified by the 'label' feature" and
   "309 .csv files". We have the 309-file layout. The paper's 46-feature table matches the
   combined/merged release, which is not on this machine.

## Update 2026-11-20: MERGED_CSV checked, still 39 features

A `MERGED_CSV` copy (63 files `Merged01–63.csv`, 8.7 GB, ~45.0M rows) was uploaded to
`/dev/shm/saksham/ciciot2023_merged/MERGED_CSV/`. All 63 files share one header: **the same 39
features plus `Label`** (40 columns). It has none of flow_duration, Srate, Drate, urg_count, Magnitue,
Radius, Covariance or Weight. So this merged release is the 39-feature version, not the 46-feature
table described in the paper. The copy also has 9 cut-off files (Merged42, 44, 46–52): each ends
partway through a row.

## Can the missing features be derived?

Not reliably. `flow_duration`, `Srate`, `Drate`, `urg_count`, `Magnitue`, `Radius`, `Covariance` and
`Weight` are computed from per-packet and per-direction data in the authors' feature extractor.
They can't be rebuilt from our per-window aggregates without the original pcaps.

## How to close the gap

- Download the combined 46-feature CSV release (`MERGED_CSV` / 169 files, `label` column) from
  <https://www.unb.ca/cic/datasets/iotdataset-2023.html>, then rerun the v4 plan
  (80/20 shuffled split, seed 42, 5% of train for validation, RF-200 + XGBoost).
- Or re-extract features from the original pcaps with the authors' scripts.

Until then, our results compare **39 features against the paper's 46**, so the paper had 8 extra
inputs available.

## Update 2026-11-20: v4 run on a 46-feature subset

The 46-feature comparison was later run on the Kaggle subset `himadri07/ciciot2023` (7.85M rows), which has all 46
features. See `results/v4_46feat/` and the v4 section of the main README. The 8 extra features add only
+0.012–0.013 macro-F1.
