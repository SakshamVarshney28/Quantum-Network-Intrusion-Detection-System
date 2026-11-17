# ============================================================
# STAGE 2 - ATTACK CLASSIFIER TRAINING SCRIPT (multi-dataset)
#
# Trains the classical multi-class model that runs AFTER the Hybrid QNN
# (Stage 1) has already flagged a flow as ATTACK. Merges as many of the
# three supported datasets as you provide, computes the same 11 flow
# features app.py uses (protocol, src_port, dst_port, flow_duration,
# total_packets, total_bytes, min/max/avg_packet_length,
# packets_per_second, bytes_per_second) from EACH dataset's own raw
# columns, normalizes attack labels so the same attack type from
# different datasets collapses into one class, and saves a SINGLE
# Stage 2 model covering the union of everything it saw.
#
# ------------------------------------------------------------
# WHERE TO PUT YOUR FILES
# ------------------------------------------------------------
#   data/nf_uq_nids_v2.csv     <- NF-UQ-NIDS-v2
#   data/cic_ddos2019.csv      <- CIC-DDoS2019 (see note below if you
#                                  have the original per-attack CSVs)
#   data/lycos_unicas_ids2018.csv <- LycoS-Unicas-IDS2018
#
# CIC-DDoS2019 is normally distributed as several separate per-attack
# CSVs (DrDoS_DNS.csv, DrDoS_LDAP.csv, Syn.csv, ...). You can either:
#   (a) concatenate them yourself into one data/cic_ddos2019.csv, or
#   (b) pass --cicddos2019 once per file:
#         --cicddos2019 data/cicddos2019/DrDoS_DNS.csv
#         --cicddos2019 data/cicddos2019/Syn.csv  ...
# Same applies to --nfuqnidsv2 / --lycosids if yours are split into
# multiple files -- just repeat the flag.
#
# ------------------------------------------------------------
# EXACT COMMAND TO RUN
# ------------------------------------------------------------
#   python train_attack_classifier.py \
#       --nfuqnidsv2 data/nf_uq_nids_v2.csv \
#       --cicddos2019 data/cic_ddos2019.csv \
#       --lycosids data/lycos_ids2017.csv
#
# ------------------------------------------------------------
# MEMORY BEHAVIOR (large datasets)
# ------------------------------------------------------------
# These datasets are large (NF-UQ-NIDS-v2 and CIC-DDoS2019 in
# particular can be tens of millions of rows / multi-GB files), so
# every CSV is read with pd.read_csv(chunksize=...) -- only one chunk
# of the RAW file is ever in memory at a time.
#
# Each chunk is immediately narrowed down to the 11 feature columns and
# appended to a small on-disk staging file (not a growing Python list),
# so RAM usage while scanning a file stays flat -- roughly the size of
# one chunk -- no matter how many gigabytes the source CSV is or how
# high --max-rows-per-dataset is set. The staged rows are stored as
# float32 (half the memory of pandas' default float64) and only read
# back into memory once, after scanning finishes, capped at whichever
# is smaller: the file's real row count or --max-rows-per-dataset.
#
# --max-rows-per-dataset is what bounds the FINAL merged training set
# that gets fully materialized for scaling/training (that part of the
# pipeline is unchanged and still needs the data in memory). If you
# don't pass it, the whole cleaned dataset is used -- fine for
# moderately sized files, but for the full uncapped NF-UQ-NIDS-v2 or
# CIC-DDoS2019 you will likely still want a cap:
#   --max-rows-per-dataset 300000
# --chunk-size lets you tune how much of the raw file is read at once
# (lower it if you're on a machine with very little RAM, e.g. 2-4 GB):
#   --chunk-size 50000
#
# Outputs (created relative to the current directory, matching the
# folder layout app.py already expects):
#   models/attack_classifier.pth   -- trained weights (ONE model)
#   models/attack_classes.json     -- ordered list of class names the
#                                      model's output neurons correspond to
#   processed/stage2_scaler.pkl    -- StandardScaler fit on the merged,
#                                      attack-only data
# ============================================================

import argparse
import gc
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from attack_types import ATTACK_CLASSES, FEATURE_COLUMNS, normalize_label
from attack_classifier_model import AttackClassifier

MODEL_OUT = os.path.join("models", "attack_classifier.pth")
CLASSES_OUT = os.path.join("models", "attack_classes.json")
SCALER_OUT = os.path.join("processed", "stage2_scaler.pkl")


# ============================================================
# COLUMN LOOKUP HELPERS
# ============================================================

def _col_map(df):
    """case/whitespace-insensitive column lookup map for this dataframe"""
    return {c.strip().lower(): c for c in df.columns}


def _find(df, col_map, *candidates):
    for cand in candidates:
        key = cand.strip().lower()
        if key in col_map:
            return df[col_map[key]]
    return None


def _require(fields, dataset_name, available_cols):
    missing = [name for name, series in fields.items() if series is None]
    if missing:
        raise SystemExit(
            f"[{dataset_name}] Could not find columns for: {missing}\n"
            f"Available columns in this file: {list(available_cols)}\n"
            "Open the CSV header and adjust the candidate names in the "
            "matching adapter function in this script."
        )


def _safe_div(numer, denom):
    denom = denom.replace(0, np.nan)
    return (numer / denom).fillna(0)


# ============================================================
# DATASET ADAPTERS
# Each returns a DataFrame with exactly FEATURE_COLUMNS + "_raw_label"
# + "_source_dataset".
# ============================================================

def adapt_nf_uq_nids_v2(df, dataset_name="NF-UQ-NIDS-v2"):
    """NetFlow-style schema (NF-UQ-NIDS-v2 / NetFlow v9-derived)."""
    cm = _col_map(df)

    protocol = _find(df, cm, "PROTOCOL")
    src_port = _find(df, cm, "L4_SRC_PORT", "SRC_PORT")
    dst_port = _find(df, cm, "L4_DST_PORT", "DST_PORT")
    duration = _find(df, cm, "FLOW_DURATION_MILLISECONDS", "FLOW_DURATION_MS")
    in_bytes = _find(df, cm, "IN_BYTES")
    out_bytes = _find(df, cm, "OUT_BYTES")
    in_pkts = _find(df, cm, "IN_PKTS")
    out_pkts = _find(df, cm, "OUT_PKTS")
    min_len = _find(df, cm, "MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT")
    max_len = _find(df, cm, "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT")
    # Prefer the multi-class "Attack" category column over the binary
    # "Label" column -- Stage 2 needs the attack family, not 0/1.
    label = _find(df, cm, "ATTACK", "LABEL")

    _require(
        {
            "protocol": protocol, "src_port": src_port, "dst_port": dst_port,
            "duration": duration, "in_bytes": in_bytes, "out_bytes": out_bytes,
            "in_pkts": in_pkts, "out_pkts": out_pkts, "label/attack": label,
        },
        dataset_name, df.columns,
    )

    total_bytes = in_bytes.astype(float) + out_bytes.astype(float)
    total_packets = in_pkts.astype(float) + out_pkts.astype(float)
    duration_ms = duration.astype(float)
    duration_s = duration_ms / 1000.0

    out = pd.DataFrame({
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "flow_duration": duration_ms,
        "total_packets": total_packets,
        "total_bytes": total_bytes,
        "min_packet_length": (min_len if min_len is not None else pd.Series(0, index=df.index)).astype(float),
        "max_packet_length": (max_len if max_len is not None else pd.Series(0, index=df.index)).astype(float),
        "avg_packet_length": _safe_div(total_bytes, total_packets),
        "packets_per_second": _safe_div(total_packets, duration_s),
        "bytes_per_second": _safe_div(total_bytes, duration_s),
    })
    out["_raw_label"] = label.astype(str)
    out["_source_dataset"] = dataset_name
    return out


def adapt_cicflowmeter_style(df, dataset_name):
    """CICFlowMeter-style schema -- used for CIC-DDoS2019. (LycoS-Unicas-IDS2018
    uses its own adapt_lycos_unicas_ids2018() below -- different exporter,
    different column names.)"""
    cm = _col_map(df)

    protocol = _find(df, cm, "Protocol")
    src_port = _find(df, cm, "Source Port", "Src Port")
    dst_port = _find(df, cm, "Destination Port", "Dst Port")
    duration_us = _find(df, cm, "Flow Duration")
    fwd_pkts = _find(df, cm, "Total Fwd Packets", "Tot Fwd Pkts")
    bwd_pkts = _find(df, cm, "Total Backward Packets", "Tot Bwd Pkts")
    fwd_bytes = _find(df, cm, "Total Length of Fwd Packets", "TotLen Fwd Pkts")
    bwd_bytes = _find(df, cm, "Total Length of Bwd Packets", "TotLen Bwd Pkts")
    label = _find(df, cm, "Label")

    _require(
        {
            "protocol": protocol, "src_port": src_port, "dst_port": dst_port,
            "duration": duration_us, "fwd_pkts": fwd_pkts, "bwd_pkts": bwd_pkts,
            "fwd_bytes": fwd_bytes, "bwd_bytes": bwd_bytes, "label": label,
        },
        dataset_name, df.columns,
    )

    total_packets = fwd_pkts.astype(float) + bwd_pkts.astype(float)
    total_bytes = fwd_bytes.astype(float) + bwd_bytes.astype(float)
    duration_ms = duration_us.astype(float) / 1000.0  # CICFlowMeter reports microseconds
    duration_s = duration_ms / 1000.0

    min_len = _find(df, cm, "Min Packet Length", "Pkt Len Min")
    max_len = _find(df, cm, "Max Packet Length", "Pkt Len Max")
    if min_len is None:
        fwd_min = _find(df, cm, "Fwd Packet Length Min", "Fwd Pkt Len Min")
        bwd_min = _find(df, cm, "Bwd Packet Length Min", "Bwd Pkt Len Min")
        min_len = pd.concat([fwd_min, bwd_min], axis=1).min(axis=1) if fwd_min is not None and bwd_min is not None else pd.Series(0, index=df.index)
    if max_len is None:
        fwd_max = _find(df, cm, "Fwd Packet Length Max", "Fwd Pkt Len Max")
        bwd_max = _find(df, cm, "Bwd Packet Length Max", "Bwd Pkt Len Max")
        max_len = pd.concat([fwd_max, bwd_max], axis=1).max(axis=1) if fwd_max is not None and bwd_max is not None else pd.Series(0, index=df.index)

    mean_len = _find(df, cm, "Average Packet Size", "Packet Length Mean", "Pkt Len Mean")
    avg_len = mean_len.astype(float) if mean_len is not None else _safe_div(total_bytes, total_packets)

    pps = _find(df, cm, "Flow Packets/s", "Flow Pkts/s")
    bps = _find(df, cm, "Flow Bytes/s", "Flow Byts/s")
    pps = pps.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0) if pps is not None else _safe_div(total_packets, duration_s)
    bps = bps.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0) if bps is not None else _safe_div(total_bytes, duration_s)

    out = pd.DataFrame({
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "flow_duration": duration_ms,
        "total_packets": total_packets,
        "total_bytes": total_bytes,
        "min_packet_length": min_len.astype(float),
        "max_packet_length": max_len.astype(float),
        "avg_packet_length": avg_len,
        "packets_per_second": pps,
        "bytes_per_second": bps,
    })
    out["_raw_label"] = label.astype(str)
    out["_source_dataset"] = dataset_name
    return out


def adapt_lycos_unicas_ids2018(df, dataset_name="LycoS-Unicas-IDS2018"):
    """LycoS-Unicas-IDS2018 schema.

    This is NOT the same column layout as adapt_cicflowmeter_style() --
    it's a different flow exporter (lowercase snake_case names, and it
    already provides global min/max/mean packet length and pkts/s,
    bytes/s directly instead of needing them derived from fwd/bwd
    totals). Kept as its own function so CIC-DDoS2019's adapter above is
    never touched by LycoS-specific column-name changes.
    """
    cm = _col_map(df)

    protocol = _find(df, cm, "ip_prot")
    dst_port = _find(df, cm, "dst_port")
    duration_raw = _find(df, cm, "flow_duration")
    fwd_pkts = _find(df, cm, "fwd_pkt_cnt")
    bwd_pkts = _find(df, cm, "bwd_pkt_cnt")
    fwd_bytes = _find(df, cm, "fwd_pkt_len_tot")
    bwd_bytes = _find(df, cm, "bwd_pkt_len_tot")
    min_len = _find(df, cm, "pkt_len_min")
    max_len = _find(df, cm, "pkt_len_max")
    mean_len = _find(df, cm, "pkt_len_mean")
    pps = _find(df, cm, "pkt_per_s")
    bps = _find(df, cm, "bytes_per_s")
    label = _find(df, cm, "label")

    _require(
        {
            "protocol": protocol, "dst_port": dst_port, "duration": duration_raw,
            "fwd_pkts": fwd_pkts, "bwd_pkts": bwd_pkts, "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes, "min_len": min_len, "max_len": max_len,
            "mean_len": mean_len, "pps": pps, "bps": bps, "label": label,
        },
        dataset_name, df.columns,
    )

    # This dataset's header has no source-port column at all (only
    # dst_port) -- default to 0 rather than failing the whole adapter
    # over one feature the exporter never captured. app.py's live
    # capture always has a real src_port, so this only affects rows
    # coming from this training source.
    src_port = pd.Series(0, index=df.index)

    total_packets = fwd_pkts.astype(float) + bwd_pkts.astype(float)
    total_bytes = fwd_bytes.astype(float) + bwd_bytes.astype(float)
    # Same exporter lineage as CICFlowMeter -> flow_duration is
    # microseconds, matching the assumption already made in
    # adapt_cicflowmeter_style() above.
    duration_ms = duration_raw.astype(float) / 1000.0

    pps = pps.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    bps = bps.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)

    out = pd.DataFrame({
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "flow_duration": duration_ms,
        "total_packets": total_packets,
        "total_bytes": total_bytes,
        "min_packet_length": min_len.astype(float),
        "max_packet_length": max_len.astype(float),
        # Exporter already computed the mean directly -- use it as-is
        # rather than re-deriving from total_bytes/total_packets.
        "avg_packet_length": mean_len.astype(float),
        "packets_per_second": pps,
        "bytes_per_second": bps,
    })
    out["_raw_label"] = label.astype(str)
    out["_source_dataset"] = dataset_name
    return out


def adapt_generic(df, label_col, dataset_name):
    """For a CSV that already has the 11 FEATURE_COLUMNS directly
    (e.g. output from this project's own preprocessing.py)."""
    cm = _col_map(df)
    missing = [c for c in FEATURE_COLUMNS if c.lower() not in cm]
    if missing:
        raise SystemExit(f"[{dataset_name}] Missing feature columns: {missing}")
    label = _find(df, cm, label_col)
    if label is None:
        raise SystemExit(f"[{dataset_name}] Label column '{label_col}' not found. Available: {list(df.columns)}")

    out = df[[cm[c.lower()] for c in FEATURE_COLUMNS]].copy()
    out.columns = FEATURE_COLUMNS
    out["_raw_label"] = label.astype(str)
    out["_source_dataset"] = dataset_name
    return out


# ============================================================
# LOADING
# ============================================================

# Rows per chunk when streaming a CSV through pd.read_csv(chunksize=...).
# This is what keeps a 13+ GB file from ever being fully materialized in
# memory -- each chunk is parsed, adapted down to the 11 narrow feature
# columns, staged to disk, and then garbage-collected before the next
# chunk is read. Tune down with --chunk-size on very memory-constrained
# machines.
CHUNK_SIZE = 250_000

# Every ATTACK_TIME chunks, force a gc pass so freed chunk memory is
# handed back to the OS promptly instead of waiting on Python's own
# collection cadence -- matters most on the huge, tens-of-millions-of-row
# files this project targets.
GC_EVERY_N_CHUNKS = 10


def load_one(path, adapter_fn, dataset_name, max_rows, chunk_size=CHUNK_SIZE):
    """Stream `path` through pd.read_csv(chunksize=...), adapt each chunk
    down to FEATURE_COLUMNS, and stage the *adapted* rows to a small
    on-disk CSV as they're produced -- instead of accumulating them as a
    growing list of DataFrames in RAM. Peak memory while scanning stays
    roughly flat (about one chunk's worth) no matter how large the source
    file is or how high max_rows is set.

    The staged rows are read back into memory exactly once, after
    scanning finishes, capped at max_rows if one was given -- that single
    in-memory DataFrame is what the rest of the (unchanged) pipeline
    works with.
    """
    print(f"Loading {dataset_name}: {path}")
    if max_rows:
        print(f"  (capping at {max_rows:,} rows -- will stop reading the file early once reached)")

    total_raw_seen = 0
    total_kept = 0
    dropped_total = 0
    wrote_header = False

    staging_fd, staging_path = tempfile.mkstemp(
        prefix=f"stage2_{dataset_name.replace(' ', '_').replace('/', '_')}_",
        suffix=".csv",
    )
    os.close(staging_fd)

    try:
        reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
        for chunk_num, chunk in enumerate(reader, start=1):
            total_raw_seen += len(chunk)

            adapted_chunk = adapter_fn(chunk)
            del chunk  # the wide raw chunk is no longer needed once adapted

            adapted_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            before = len(adapted_chunk)
            adapted_chunk.dropna(subset=FEATURE_COLUMNS, inplace=True)
            dropped_total += before - len(adapted_chunk)

            if max_rows is not None:
                remaining = max_rows - total_kept
                if remaining <= 0:
                    del adapted_chunk
                    break
                if len(adapted_chunk) > remaining:
                    adapted_chunk = adapted_chunk.sample(n=remaining, random_state=42)

            if len(adapted_chunk):
                # float32 instead of pandas' default float64 halves the
                # memory (and staging-disk) footprint of the 11 numeric
                # feature columns -- adds up fast once millions of rows
                # are staged. Training still converts to torch.float32
                # later regardless, so this changes nothing downstream.
                adapted_chunk[FEATURE_COLUMNS] = adapted_chunk[FEATURE_COLUMNS].astype(np.float32)

                adapted_chunk.to_csv(
                    staging_path, mode="a", index=False, header=not wrote_header
                )
                wrote_header = True
                total_kept += len(adapted_chunk)

            del adapted_chunk

            if max_rows is not None and total_kept >= max_rows:
                break

            if chunk_num % GC_EVERY_N_CHUNKS == 0:
                gc.collect()

        if total_kept == 0:
            raise SystemExit(
                f"[{dataset_name}] No usable rows found in {path} -- every row was either "
                "invalid/missing a required feature, or --max-rows-per-dataset was 0."
            )

        print(f"  scanned {total_raw_seen:,} raw rows" + (" (stopped early once cap was reached)" if max_rows and total_raw_seen and total_kept >= max_rows else ""))
        if dropped_total:
            print(f"  dropped {dropped_total:,} rows with invalid/missing feature values")
        print(f"  usable: {total_kept:,} rows")

        # Single bounded read back off disk -- bounded because total_kept
        # (and therefore the staging file) is capped at max_rows whenever
        # one is given, which is the normal way to run this on the
        # full-size datasets. Explicit dtypes avoid a float64 upcast on
        # the way back in, and the label columns become categoricals
        # (cheap for the handful of distinct attack-type strings repeated
        # over potentially millions of rows).
        dtype_map = {c: np.float32 for c in FEATURE_COLUMNS}
        adapted = pd.read_csv(staging_path, dtype=dtype_map)
        adapted["_raw_label"] = adapted["_raw_label"].astype("category")
        adapted["_source_dataset"] = adapted["_source_dataset"].astype("category")
        return adapted
    finally:
        try:
            os.remove(staging_path)
        except OSError:
            pass


def load_many(paths, adapter_fn, dataset_name, max_rows, chunk_size=CHUNK_SIZE):
    if not paths:
        return None
    parts = [load_one(p, adapter_fn, dataset_name, max_rows, chunk_size) for p in paths]
    merged = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train the Stage 2 attack-type classifier on one or more datasets"
    )
    parser.add_argument("--nfuqnidsv2", action="append", default=[], metavar="CSV",
                         help="Path to an NF-UQ-NIDS-v2 CSV (repeatable)")
    parser.add_argument("--cicddos2019", action="append", default=[], metavar="CSV",
                         help="Path to a CIC-DDoS2019 CSV (repeatable)")
    parser.add_argument("--lycosids", action="append", default=[], metavar="CSV",
                         help="Path to a LycoS-Unicas-IDS2018 CSV (repeatable)")
    parser.add_argument("--data", action="append", default=[], metavar="CSV",
                         help="Path to a generic CSV that already has the 11 "
                              "FEATURE_COLUMNS directly (repeatable)")
    parser.add_argument("--label-col", default=None,
                         help="Label column name for --data files (auto-detected if omitted)")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None,
                         help="Subsample each dataset to at most this many rows after cleaning")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                         help=f"Rows read from each source CSV at a time (default: {CHUNK_SIZE:,}). "
                              "Lower this on machines with very little RAM.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=15,
                         help="Stop training if validation accuracy doesn't improve for this "
                              "many consecutive epochs (default: 15)")
    parser.add_argument("--lr-patience", type=int, default=5,
                         help="Epochs of no validation-accuracy improvement before "
                              "ReduceLROnPlateau cuts the learning rate (default: 5)")
    parser.add_argument("--lr-factor", type=float, default=0.5,
                         help="Multiply learning rate by this factor when the scheduler "
                              "triggers (default: 0.5)")
    args = parser.parse_args()

    if not any([args.nfuqnidsv2, args.cicddos2019, args.lycosids, args.data]):
        raise SystemExit(
            "No datasets given. Pass at least one of --nfuqnidsv2 / --cicddos2019 / "
            "--lycosids / --data. See the header of this script for the exact command."
        )

    if args.max_rows_per_dataset is None:
        print(
            "NOTE: --max-rows-per-dataset was not set, so the full cleaned size of "
            "each dataset will be used. Loading itself streams safely regardless of "
            "file size, but if the final merged training set ends up too large to "
            "fit in memory for scaling/training, re-run with e.g. "
            "--max-rows-per-dataset 300000."
        )

    frames = []

    part = load_many(args.nfuqnidsv2, adapt_nf_uq_nids_v2, "NF-UQ-NIDS-v2",
                      args.max_rows_per_dataset, args.chunk_size)
    if part is not None:
        frames.append(part)

    part = load_many(
        args.cicddos2019,
        lambda d: adapt_cicflowmeter_style(d, "CIC-DDoS2019"),
        "CIC-DDoS2019", args.max_rows_per_dataset, args.chunk_size,
    )
    if part is not None:
        frames.append(part)

    part = load_many(
        args.lycosids,
        lambda d: adapt_lycos_unicas_ids2018(d, "LycoS-Unicas-IDS2018"),
        "LycoS-Unicas-IDS2018", args.max_rows_per_dataset, args.chunk_size,
    )
    if part is not None:
        frames.append(part)

    if args.data:
        label_col = args.label_col
        if not label_col:
            raise SystemExit("--label-col is required when using --data")
        part = load_many(
            args.data,
            lambda d: adapt_generic(d, label_col, "Custom"),
            "Custom", args.max_rows_per_dataset, args.chunk_size,
        )
        if part is not None:
            frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"\nMerged dataset: {len(df):,} rows total from {df['_source_dataset'].nunique()} source(s)")
    print(df.groupby("_source_dataset", observed=True).size().to_string())

    # ---- Normalize labels, drop benign rows, dedupe classes across datasets ----
    normalized = df["_raw_label"].astype(str).map(normalize_label)
    benign_mask = normalized.isna()
    print(f"\nDropping {int(benign_mask.sum()):,} benign/normal rows (Stage 1's job, not Stage 2's).")
    df = df[~benign_mask].copy()
    normalized = normalized[~benign_mask]
    df["_attack_label"] = normalized.values
    del normalized, benign_mask
    gc.collect()

    unmapped = sorted(set(df["_attack_label"]) - set(ATTACK_CLASSES))
    if unmapped:
        print(
            "\nNOTE: these labels didn't match a known alias and were kept as their "
            "own class (add aliases in attack_types.py if you'd rather merge them "
            f"into one of the 16 canonical types): {unmapped}"
        )

    # ---- Features / labels ----
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df["_attack_label"]
    del df
    gc.collect()

    if len(X) == 0:
        raise SystemExit("No valid attack rows left after cleaning. Check your dataset/label columns.")

    present = set(y_raw.unique())
    classes = [c for c in ATTACK_CLASSES if c in present] + sorted(present - set(ATTACK_CLASSES))
    class_to_idx = {name: i for i, name in enumerate(classes)}

    print(f"\nFinal merged training set: {len(X):,} attack rows across {len(classes)} classes:")
    for name in classes:
        print(f"  {name:24s} {int((y_raw == name).sum()):>10,}")

    y = y_raw.map(class_to_idx).values

    # ---- Scale, split ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    del X
    gc.collect()

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=42, stratify=stratify
    )
    del X_scaled
    gc.collect()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # ---- Class weights (used ONLY for the sampler below) ----
    # Balancing is applied in exactly one place: WeightedRandomSampler
    # oversamples minority classes when building each epoch's batches.
    # The loss stays a plain, unweighted CrossEntropyLoss -- applying
    # inverse-frequency weighting in BOTH the sampler and the loss
    # compounds to roughly the weight squared for rare classes, which is
    # what caused the previous run's collapse (val_acc near 1/n_classes,
    # 0% on multiple classes): gradients for rare-class batches were
    # amplified twice over and blew up training.
    n_classes = len(classes)
    train_class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    train_class_counts[train_class_counts == 0] = 1.0  # guard divide-by-zero for any class
    # missing entirely from this particular train split
    class_weights = len(y_train) / (n_classes * train_class_counts)

    # ---- Train ----
    model = AttackClassifier(input_dim=len(FEATURE_COLUMNS), num_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
    )

    # WeightedRandomSampler draws training rows in proportion to their
    # class's inverse frequency, so minority attack classes (e.g. rare
    # ones like Heartbleed) show up in roughly the same numbers per batch
    # as majority ones like DDoS, instead of being drowned out.
    sample_weights = torch.tensor(class_weights[y_train], dtype=torch.double)
    n_train = len(X_train_t)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=n_train, replacement=True
    )

    print(f"\nTraining for up to {args.epochs} epochs on {n_train:,} rows "
          f"(holding out {len(X_test_t):,} for evaluation), "
          f"early stopping patience={args.early_stop_patience}...")

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Re-drawing from the sampler each epoch re-samples (with
        # replacement) fresh balanced batch composition and order --
        # equivalent to a balanced-shuffle replacement for the old
        # torch.randperm(n_train) call.
        perm = torch.tensor(list(sampler), dtype=torch.long)
        total_loss = 0.0

        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            if len(idx) < 2:
                # BatchNorm1d can't compute batch statistics from a single
                # sample in training mode -- skip a stray trailing batch of
                # size 1 rather than crash (loses at most 1 row/epoch).
                continue
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            test_preds = torch.argmax(model(X_test_t), dim=1)
            val_acc = (test_preds == y_test_t).float().mean().item()

        scheduler.step(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == args.epochs or improved:
            current_lr = optimizer.param_groups[0]["lr"]
            marker = "  (best)" if improved else ""
            print(f"  epoch {epoch:3d}/{args.epochs}  train_loss={total_loss / n_train:.4f}  "
                  f"val_acc={val_acc:.4f}  lr={current_lr:.2e}{marker}")

        if epochs_no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}: no val_acc improvement for "
                  f"{args.early_stop_patience} epochs (best={best_val_acc:.4f} at epoch {best_epoch}).")
            break

    # ---- Restore best-validation-accuracy weights (not necessarily the last epoch) ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nUsing best model from epoch {best_epoch} (val_acc={best_val_acc:.4f}) "
              "for final report and saved artifacts.")

    # ---- Final per-class report ----
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_test_t), dim=1).numpy()
    print("\nPer-class test accuracy:")
    y_test_np = y_test_t.numpy()
    for name, idx in class_to_idx.items():
        mask = y_test_np == idx
        if mask.sum() == 0:
            continue
        class_acc = (test_preds[mask] == idx).mean()
        print(f"  {name:24s} {class_acc:.4f}  (n={int(mask.sum())})")

    # ---- Save artifacts (ONE model covering everything merged above) ----
    os.makedirs("models", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    torch.save(model.state_dict(), MODEL_OUT)
    with open(CLASSES_OUT, "w") as f:
        json.dump(classes, f, indent=2)
    joblib.dump(scaler, SCALER_OUT)

    print(f"\nSaved model  -> {MODEL_OUT}")
    print(f"Saved classes-> {CLASSES_OUT}  ({len(classes)} classes)")
    print(f"Saved scaler -> {SCALER_OUT}")
    print("\nDone. Restart the app to pick up the new Stage 2 model.")


if __name__ == "__main__":
    main()