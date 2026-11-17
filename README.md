# Hybrid Quantum-Classical Network Intrusion Detection & Attack Classification System

A two-stage network-security application that detects suspicious network
flows and classifies likely attack types.

## What the project does

The system converts each network flow into 11 numerical features:

```text
protocol
src_port
dst_port
flow_duration
total_packets
total_bytes
min_packet_length
max_packet_length
avg_packet_length
packets_per_second
bytes_per_second
```

It then uses two stages:

```text
Stage 1: Hybrid Quantum Neural Network
         Normal traffic vs Attack traffic

Stage 2: Classical attack classifier
         DDoS, Port Scan, Brute Force, XSS, and other attack types
```

The application supports:

- CSV network-flow analysis
- Real-time IPv4 packet capture with Scapy
- Normal/attack prediction
- Attack-type classification
- Confidence and severity display
- Threat-level visualization
- Suspicious-IP tracking
- Training-loss graph
- Confusion matrix
- Classification report
- Analysis history
- CSV/PDF reports
- Tkinter desktop interface
- Streamlit browser interface

## Why a hybrid quantum model?

The classical neural-network layers first reduce the 11 flow features to four
values. These values are encoded into four simulated qubits using PennyLane.
Trainable quantum rotations and entangling operations transform the values,
and four measured expectation values are passed to classical layers for the
final normal/attack decision.

The quantum circuit is therefore a trainable feature-transformation layer. It
does not capture packets itself and this project does not claim automatic
quantum advantage. The current implementation uses PennyLane's
`default.qubit` simulator, so no quantum hardware is required.

## Datasets used

The project is designed to combine network-flow data from:

1. **NF-UQ-NIDS-v2**
   - Network intrusion-detection flow dataset.
   - Source: <https://www.kaggle.com/datasets/madhavmalhotra/nf-uq-nids-v2>

2. **LycoS-Unicas-IDS2018**
   - Intrusion-detection flow data derived from the CIC-IDS2018 family.
   - Use the dataset source supplied by the project supervisor or the
     corresponding LycoS/Unicas release.

3. **CIC-DDoS2019**
   - Distributed-denial-of-service traffic dataset.
   - Official project page:
     <https://www.unb.ca/cic/datasets/ddos-2019.html>

Raw datasets are intentionally not included in this repository. Dataset
licenses and file sizes vary, so download them from their official or
authorized sources and keep them locally.

## Local folder layout

After obtaining the data, the local workspace should contain:

```text
datasets/
├── NF-UQ-NIDS-v2.csv
├── LycoS-Unicas-IDS2018.csv
├── 01-12/
└── 03-11/
```

The exact filenames and source folders depend on the downloaded release.
Check the arguments accepted by `combine_datasets.py` and
`train_attack_classifier.py` if a release uses different names.

## Installation

Use Python 3.11 or another version supported by the installed dependencies.

```bash
git clone https://github.com/shibyte/Quantum_Network_Intrusion_Detection_System.git
cd Quantum_Network_Intrusion_Detection_System

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, if Tkinter is unavailable, install a Python distribution that
includes Tk support before running the desktop application.

## Build the local training artifacts

These commands require the datasets in the local `datasets/` folder:

```bash
python combine_datasets.py
python preprocessing.py
python train_qnn.py
python test_qnn.py
python train_attack_classifier.py
```

The generated files are intentionally ignored by Git:

```text
combined_dataset.csv
processed/
models/
results/
```

## Run the applications

### Tkinter desktop application

For CSV analysis:

```bash
python app.py
```

For macOS live packet capture, administrator permission may be required:

```bash
sudo ./.venv/bin/python app.py
```

## CSV format for analysis

Uploaded CSV files must contain these 11 feature columns:

```text
protocol,src_port,dst_port,flow_duration,total_packets,total_bytes,
min_packet_length,max_packet_length,avg_packet_length,
packets_per_second,bytes_per_second
```

Optional reference columns are:

```text
label
attack_type
```

The application uses the 11 feature columns for prediction. It does not use
the reference labels as model inputs.

## Important limitations

- The live capture model analyzes flow statistics, not packet payload text.
- SQL Injection and XSS payload content cannot be directly inspected by the
  current 11-feature model.
- Live traffic can differ from benchmark datasets and may cause false
  positives or false negatives.
- The quantum circuit currently runs in simulation.
- Attack-type accuracy depends on the classes and distributions present in
  the training datasets.
- A prediction is a model result, not proof of malicious activity.

## Project files

```text
app.py                       Tkinter desktop application
qnn_model.py                 Hybrid quantum-classical model
attack_classifier_model.py   Stage 2 attack classifier
attack_types.py              Attack labels and normalization
combine_datasets.py          Dataset combination
preprocessing.py             Cleaning, splitting, and scaling
train_qnn.py                 Stage 1 training
test_qnn.py                  Stage 1 evaluation
train_attack_classifier.py   Stage 2 training
feature_selection.py         Feature utilities
requirements.txt             Python dependencies
```

