import os
import glob
import pandas as pd
import numpy as np

print("=" * 60)
print("NETWORK INTRUSION DETECTION USING HYBRID QNN")
print("=" * 60)

print("\nLoading Datasets...")

# -------------------------------------------------------
# NF-UQ-NIDS-v2
# -------------------------------------------------------

nf = pd.read_csv(
    "datasets/NF-UQ-NIDS-v2.csv",
    nrows=200000,
    low_memory=False
)

print("✅ NF Loaded")
print("Shape :", nf.shape)

# -------------------------------------------------------
# LycoS IDS2018
# -------------------------------------------------------

lycos = pd.read_csv(
    "datasets/LycoS-Unicas-IDS2018.csv",
    nrows=200000,
    low_memory=False
)

print("✅ LycoS Loaded")
print("Shape :", lycos.shape)

# -------------------------------------------------------
# CIC DDoS 2019
# -------------------------------------------------------

print("\nLoading CIC DDoS2019...")

files = glob.glob("datasets/01-12/*.csv")
files += glob.glob("datasets/03-11/*.csv")

print("Files Found :", len(files))

cic_frames = []

for file in files:

    print("Reading", os.path.basename(file))

    temp = pd.read_csv(
        file,
        nrows=15000,
        low_memory=False
    )

    cic_frames.append(temp)

cic = pd.concat(
    cic_frames,
    ignore_index=True
)

print("✅ CIC Loaded")
print("Shape :", cic.shape)

print("\n")
print("=" * 60)
print("ALL DATASETS LOADED SUCCESSFULLY")
print("=" * 60)# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

print("\nCleaning column names...")

# Remove extra spaces
nf.columns = nf.columns.str.strip()
lycos.columns = lycos.columns.str.strip()
cic.columns = cic.columns.str.strip()

# Convert all column names to lowercase
nf.columns = nf.columns.str.lower()
lycos.columns = lycos.columns.str.lower()
cic.columns = cic.columns.str.lower()

print("✅ Column names cleaned")

# ==========================================================
# STANDARDIZE LABEL COLUMN
# ==========================================================

print("\nStandardizing label column...")

# Rename label columns
if "label" not in nf.columns and "Label".lower() in nf.columns:
    nf.rename(columns={"label": "label"}, inplace=True)

if "label" not in lycos.columns:
    lycos.rename(columns={"Label": "label"}, inplace=True)

if "label" not in cic.columns:
    cic.rename(columns={"Label": "label"}, inplace=True)

print("NF Label  :", "label" in nf.columns)
print("LycoS     :", "label" in lycos.columns)
print("CIC       :", "label" in cic.columns)

# ==========================================================
# SHOW COLUMN NAMES
# ==========================================================

print("\nNF-UQ Columns")
print(nf.columns.tolist())

print("\nLycoS Columns")
print(lycos.columns.tolist())

print("\nCIC Columns")
print(cic.columns.tolist())
# ==========================================================
# CREATE UNIFIED DATASETS
# ==========================================================

print("\nCreating unified datasets...")

nf_new = pd.DataFrame()

nf_new["protocol"] = nf["protocol"]
nf_new["src_port"] = nf["l4_src_port"]
nf_new["dst_port"] = nf["l4_dst_port"]
nf_new["flow_duration"] = nf["flow_duration_milliseconds"]

nf_new["total_packets"] = nf["in_pkts"] + nf["out_pkts"]
nf_new["total_bytes"] = nf["in_bytes"] + nf["out_bytes"]

nf_new["min_packet_length"] = nf["min_ip_pkt_len"]
nf_new["max_packet_length"] = nf["max_ip_pkt_len"]

nf_new["avg_packet_length"] = (
    nf["min_ip_pkt_len"] +
    nf["max_ip_pkt_len"]
) / 2

nf_new["packets_per_second"] = (
    nf["in_pkts"] +
    nf["out_pkts"]
) / (
    nf["flow_duration_milliseconds"] + 1
)

nf_new["bytes_per_second"] = (
    nf["in_bytes"] +
    nf["out_bytes"]
) / (
    nf["flow_duration_milliseconds"] + 1
)

nf_new["label"] = nf["label"]

print("NF Unified :", nf_new.shape)# ==========================================================
# CREATE LYCOS UNIFIED DATASET
# ==========================================================

lycos_new = pd.DataFrame()

lycos_new["protocol"] = lycos["ip_prot"]
lycos_new["src_port"] = 0
lycos_new["dst_port"] = lycos["dst_port"]

lycos_new["flow_duration"] = lycos["flow_duration"]

lycos_new["total_packets"] = (
    lycos["fwd_pkt_cnt"] +
    lycos["bwd_pkt_cnt"]
)

lycos_new["total_bytes"] = (
    lycos["fwd_pkt_len_tot"] +
    lycos["bwd_pkt_len_tot"]
)

lycos_new["min_packet_length"] = lycos["pkt_len_min"]
lycos_new["max_packet_length"] = lycos["pkt_len_max"]
lycos_new["avg_packet_length"] = lycos["pkt_len_mean"]

lycos_new["packets_per_second"] = lycos["pkt_per_s"]
lycos_new["bytes_per_second"] = lycos["bytes_per_s"]

lycos_new["label"] = lycos["label"]

print("LycoS Unified :", lycos_new.shape)
# ==========================================================
# CREATE CIC UNIFIED DATASET
# ==========================================================

cic_new = pd.DataFrame()

cic_new["protocol"] = cic["protocol"]

cic_new["src_port"] = cic["source port"]
cic_new["dst_port"] = cic["destination port"]

cic_new["flow_duration"] = cic["flow duration"]

cic_new["total_packets"] = (
    cic["total fwd packets"] +
    cic["total backward packets"]
)

cic_new["total_bytes"] = (
    cic["total length of fwd packets"] +
    cic["total length of bwd packets"]
)

cic_new["min_packet_length"] = cic["min packet length"]

cic_new["max_packet_length"] = cic["max packet length"]

cic_new["avg_packet_length"] = cic["packet length mean"]

cic_new["packets_per_second"] = cic["flow packets/s"]

cic_new["bytes_per_second"] = cic["flow bytes/s"]

cic_new["label"] = cic["label"]

print("CIC Unified :", cic_new.shape)
# ==========================================================
# MERGE ALL DATASETS
# ==========================================================

print("\nMerging datasets...")

combined = pd.concat(
    [nf_new, lycos_new, cic_new],
    ignore_index=True
)

print("Merged Shape :", combined.shape)

# ==========================================================
# REMOVE MISSING VALUES
# ==========================================================

combined.replace([np.inf, -np.inf], np.nan, inplace=True)

combined.dropna(inplace=True)

combined.drop_duplicates(inplace=True)

print("After Cleaning :", combined.shape)

# ==========================================================
# CONVERT LABEL TO BINARY  +  PRESERVE ATTACK CATEGORY
# ==========================================================
#
# combined["label"] currently holds each dataset's *raw* text label
# (e.g. "Benign", "DDoS", "DrDoS_DNS", "DoS attacks-Hulk",
# "FTP-Patator", "PortScan"...). We keep that raw text around long
# enough to bucket it into a canonical attack category, then collapse
# it into the binary label the Hybrid QNN trains on. The canonical
# category is saved as a separate "attack_type" column so a second
# -stage classifier can later learn to predict the specific attack
# type, without changing what the QNN itself is trained on.
# ==========================================================

print("\nConverting labels...")

combined["label"] = combined["label"].astype(str).str.lower().str.strip()


def map_attack_category(raw_label: str) -> str:
    """Buckets a raw dataset-specific label string into a small set of
    canonical attack categories. Order matters below (e.g. 'ddos' is
    checked before the more general 'dos', since 'ddos' contains 'dos'
    as a substring)."""
    text = raw_label.lower()

    if "benign" in text or text == "0" or text == "normal":
        return "Benign"

    if "ddos" in text or "drdos" in text:
        return "DDoS"

    if "dos" in text:
        return "DoS"

    if "portscan" in text or "port scan" in text or "scanning" in text or text == "scan":
        return "PortScan"

    if "bot" in text:
        return "Bot"

    if "brute" in text or "patator" in text:
        return "BruteForce"

    if "sql" in text or "xss" in text or "web attack" in text or "injection" in text:
        return "WebAttack"

    if "infiltration" in text:
        return "Infiltration"

    if "heartbleed" in text:
        return "Heartbleed"

    # Unrecognised but non-benign text still counts as an attack;
    # bucket it so it isn't silently dropped from analysis.
    return "Other"


combined["attack_type"] = combined["label"].apply(map_attack_category)

# Binary label is now DERIVED from attack_type, so the two columns can
# never disagree with each other.
combined["label"] = combined["attack_type"].apply(lambda cat: 0 if cat == "Benign" else 1)

print("\nBinary Label Distribution")
print(combined["label"].value_counts())

print("\nAttack Type Distribution")
print(combined["attack_type"].value_counts())

print(
    "\nNOTE: if a dataset's raw label column only ever contained "
    "generic text like 'Benign'/'Attack' (no specific category), its "
    "attack rows will show up here as 'Other' rather than a named "
    "type — that's a limitation of that raw file, not of this script."
)

# ==========================================================
# SAVE DATASET
# ==========================================================

combined.to_csv(
    "combined_dataset.csv",
    index=False
)

print("\n✅ combined_dataset.csv Saved Successfully")
print("Final Shape :", combined.shape)
print("Columns     :", combined.columns.tolist())