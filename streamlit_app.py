"""Streamlit demonstration UI for the Quantum Network IDS.

Run with: ``streamlit run streamlit_app.py``
The application deliberately keeps packet capture bounded: Streamlit reruns
the script on interaction, so a permanent background sniffer would be unsafe.
"""

from __future__ import annotations

import io
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from attack_classifier_model import AttackClassifier
from qnn_model import HybridQNN

try:
    from scapy.all import IP, TCP, UDP, get_if_list, sniff

    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hybrid_qnn.pth"
SCALER_PATH = ROOT / "processed" / "scaler.pkl"
STAGE2_MODEL_PATH = ROOT / "models" / "attack_classifier.pth"
STAGE2_SCALER_PATH = ROOT / "processed" / "stage2_scaler.pkl"
STAGE2_CLASSES_PATH = ROOT / "models" / "attack_classes.json"

FEATURE_COLUMNS = [
    "protocol",
    "src_port",
    "dst_port",
    "flow_duration",
    "total_packets",
    "total_bytes",
    "min_packet_length",
    "max_packet_length",
    "avg_packet_length",
    "packets_per_second",
    "bytes_per_second",
]
BATCH_SIZE = 2000


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    """Load artifacts once per Streamlit process and retain useful errors."""
    result: dict[str, Any] = {
        "model": None, "scaler": None, "stage2_model": None,
        "stage2_scaler": None, "classes": [], "errors": [],
    }
    try:
        model = HybridQNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        result["model"] = model
    except Exception as exc:
        result["errors"].append(f"Stage 1 model: {exc}")
    try:
        result["scaler"] = joblib.load(SCALER_PATH)
    except Exception as exc:
        result["errors"].append(f"Stage 1 scaler: {exc}")
    try:
        with STAGE2_CLASSES_PATH.open(encoding="utf-8") as handle:
            classes = json.load(handle)
        if not isinstance(classes, list) or not classes:
            raise ValueError("class list is empty")
        stage2 = AttackClassifier(input_dim=len(FEATURE_COLUMNS), num_classes=len(classes))
        stage2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location="cpu"))
        stage2.eval()
        result.update(
            stage2_model=stage2,
            stage2_scaler=joblib.load(STAGE2_SCALER_PATH),
            classes=classes,
        )
    except Exception as exc:
        result["errors"].append(f"Stage 2 classifier unavailable: {exc}")
    return result


def diagnostics(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "PyTorch": torch.__version__,
        "CUDA": "Available" if torch.cuda.is_available() else "CPU only",
        "Stage 1 model": artifacts["model"] is not None and MODEL_PATH.exists(),
        "Stage 1 scaler": artifacts["scaler"] is not None and SCALER_PATH.exists(),
        "Stage 2 model": artifacts["stage2_model"] is not None,
        "Scapy": SCAPY_AVAILABLE,
    }


def _predict(features: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Return predictions for valid feature rows, preserving source row index."""
    cleaned = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("No rows contain all 11 numeric feature values.")
    if artifacts["model"] is None or artifacts["scaler"] is None:
        raise RuntimeError("Stage 1 model and scaler must both be available.")
    scaled = artifacts["scaler"].transform(cleaned.to_numpy())
    predictions: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(scaled), BATCH_SIZE):
            output = artifacts["model"](
                torch.tensor(scaled[start:start + BATCH_SIZE], dtype=torch.float32)
            )
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            predictions.extend(prediction.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())
    result = cleaned.copy()
    result["prediction"] = np.where(np.asarray(predictions) == 1, "Attack", "Normal")
    result["confidence"] = np.round(np.asarray(confidences) * 100, 2)
    result["attack_type"] = "Not applicable"
    attack_mask = result["prediction"].eq("Attack")
    if attack_mask.any() and artifacts["stage2_model"] is not None:
        stage2_scaled = artifacts["stage2_scaler"].transform(
            cleaned.loc[attack_mask].to_numpy()
        )
        with torch.inference_mode():
            output = artifacts["stage2_model"](
                torch.tensor(stage2_scaled, dtype=torch.float32)
            )
            labels = output.argmax(dim=1).cpu().numpy()
        classes = artifacts["classes"]
        result.loc[attack_mask, "attack_type"] = [
            classes[int(index)] if int(index) < len(classes) else "Unknown"
            for index in labels
        ]
    return result


def summarize(predictions: pd.DataFrame, dataset: str) -> dict[str, Any]:
    total = len(predictions)
    attack = int(predictions["prediction"].eq("Attack").sum())
    attack_pct = round(100 * attack / total, 2)
    threat = "LOW" if attack_pct == 0 else "MEDIUM" if attack_pct < 10 else "HIGH" if attack_pct < 40 else "CRITICAL"
    severity = "Critical" if threat == "CRITICAL" else "High" if threat == "HIGH" else "Medium" if threat == "MEDIUM" else "Low"
    breakdown = predictions.loc[predictions["prediction"].eq("Attack"), "attack_type"].value_counts().to_dict()
    return {
        "dataset": dataset, "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "total": total, "normal": total - attack, "attack": attack,
        "attack_pct": attack_pct, "threat": threat, "severity": severity,
        "confidence": round(float(predictions["confidence"].mean()), 2),
        "status": "ATTACK DETECTED" if attack else "NORMAL TRAFFIC",
        "breakdown": breakdown,
    }


def report_csv(summary: dict[str, Any], predictions: pd.DataFrame) -> bytes:
    output = io.StringIO()
    pd.DataFrame([summary]).to_csv(output, index=False)
    output.write("\nPrediction details\n")
    predictions.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def report_pdf(summary: dict[str, Any]) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        figure = Figure(figsize=(8.5, 11), facecolor="white")
        axis = figure.add_subplot(111)
        axis.axis("off")
        lines = [
            "QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
            "Hybrid Quantum Neural Network — Analysis Report", "",
            *(f"{key.title():16}: {value}" for key, value in summary.items()
              if key != "breakdown"),
        ]
        if summary["breakdown"]:
            lines += ["", "Attack type breakdown:"]
            lines += [f"  {name}: {count}" for name, count in summary["breakdown"].items()]
        axis.text(0.06, 0.94, "\n".join(lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(figure, bbox_inches="tight")
    return buffer.getvalue()


def capture_features(packets: list[Any]) -> pd.DataFrame:
    rows = []
    for packet in packets:
        length = len(packet)
        protocol = 6 if packet.haslayer(TCP) else 17 if packet.haslayer(UDP) else 0
        src_port = int(packet[TCP].sport) if packet.haslayer(TCP) else int(packet[UDP].sport) if packet.haslayer(UDP) else 0
        dst_port = int(packet[TCP].dport) if packet.haslayer(TCP) else int(packet[UDP].dport) if packet.haslayer(UDP) else 0
        rows.append([protocol, src_port, dst_port, 0.0, 1, length, length, length, float(length), 1.0, float(length)])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def style_app() -> None:
    st.markdown(
        """<style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        [data-testid="stMetric"] {background: #f7f5ff; border: 1px solid #e4ddff;
        padding: 1rem; border-radius: 12px;}
        </style>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Quantum IDS", page_icon="🛡️", layout="wide")
    style_app()
    artifacts = load_artifacts()
    with st.sidebar:
        st.title("🛡️ Quantum IDS")
        st.caption("Hybrid QNN network defense console")
        page = st.radio("Navigate", ["Dashboard", "CSV Analysis", "Live Capture", "Diagnostics"])
        st.divider()
        st.caption("Artifacts are loaded from the repository at runtime.")
        if artifacts["model"] is not None and artifacts["scaler"] is not None:
            st.success("Stage 1 ready")
        else:
            st.error("Stage 1 unavailable")
        st.info("Stage 2 ready" if artifacts["stage2_model"] is not None else "Stage 2 unavailable")

    st.title("Quantum Network Intrusion Detection System")
    st.caption("Evidence-based traffic analysis using the repository's trained artifacts — no simulated results.")
    if artifacts["errors"] and page != "Diagnostics":
        with st.expander("Artifact notices", expanded=False):
            for error in artifacts["errors"]:
                st.warning(error)

    if page == "Diagnostics":
        st.header("System diagnostics")
        info = diagnostics(artifacts)
        cols = st.columns(4)
        for column, (name, value) in zip(cols, info.items()):
            column.metric(name, "Ready" if value is True else "Unavailable" if value is False else str(value))
        st.subheader("Artifact inventory")
        st.dataframe(pd.DataFrame([
            {"Artifact": str(path.relative_to(ROOT)), "Present": path.exists()}
            for path in [MODEL_PATH, SCALER_PATH, STAGE2_MODEL_PATH, STAGE2_SCALER_PATH, STAGE2_CLASSES_PATH]
        ]), hide_index=True, use_container_width=True)
        if not SCAPY_AVAILABLE:
            st.warning("Scapy is not importable. Install it and grant packet-capture permissions to enable live capture.")
        return

    predictions = st.session_state.get("predictions")
    summary = st.session_state.get("summary")
    if page == "Dashboard":
        st.header("Security operations dashboard")
        if summary is None:
            st.info("Upload a compatible CSV or run a bounded capture to populate the dashboard with real predictions.")
            st.subheader("Pipeline")
            st.write("CSV / packets → 11-feature validation → StandardScaler → Hybrid QNN → optional Stage 2 attack classification")
            return
        cols = st.columns(5)
        for column, label, value in zip(cols, ["Flows", "Normal", "Attacks", "Attack rate", "Threat"], [summary["total"], summary["normal"], summary["attack"], f'{summary["attack_pct"]}%', summary["threat"]]):
            column.metric(label, value)
        st.success(f'{summary["status"]} · average confidence {summary["confidence"]}%')
    elif page == "CSV Analysis":
        st.header("CSV analysis")
        st.write("Required columns (case-insensitive): " + ", ".join(FEATURE_COLUMNS))
        uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])
        if uploaded is not None:
            try:
                source = pd.read_csv(uploaded)
                normalized = source.rename(columns={column: str(column).strip().lower() for column in source.columns})
                missing = [column for column in FEATURE_COLUMNS if column not in normalized.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                elif st.button("Analyze dataset", type="primary"):
                    with st.spinner("Running the trained models..."):
                        result = _predict(normalized, artifacts)
                    st.session_state.predictions = result
                    st.session_state.summary = summarize(result, uploaded.name)
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not analyze this file: {exc}")
        if predictions is not None and summary is not None:
            st.subheader("Analysis report")
            st.write(f'**{summary["status"]}** · {summary["severity"]} severity · {summary["confidence"]}% average confidence')
            st.dataframe(predictions.head(1000), use_container_width=True, hide_index=True)
    elif page == "Live Capture":
        st.header("Bounded live packet capture")
        st.warning("Requires administrator/root capture permission. This action captures only a bounded batch; it never starts a background sniffer.")
        if not SCAPY_AVAILABLE:
            st.error("Scapy is unavailable in this environment.")
        else:
            interfaces = ["default"] + list(get_if_list())
            interface = st.selectbox("Interface", interfaces)
            count = st.slider("Packets", 1, 100, 20)
            timeout = st.slider("Timeout (seconds)", 1, 15, 5)
            if st.button("Capture and analyze", type="primary"):
                try:
                    kwargs: dict[str, Any] = {"count": count, "timeout": timeout, "store": True}
                    if interface != "default":
                        kwargs["iface"] = interface
                    with st.spinner("Capturing packets..."):
                        packets = sniff(**kwargs)
                    if not packets:
                        st.warning("No packets captured. Check the interface and capture permissions.")
                    else:
                        result = _predict(capture_features(list(packets)), artifacts)
                        st.session_state.predictions = result
                        st.session_state.summary = summarize(result, f"Live capture ({len(packets)} packets)")
                        st.success(f"Captured and analyzed {len(packets)} packets.")
                        st.rerun()
                except PermissionError:
                    st.error("Permission denied. Run with the OS privileges required by your packet-capture backend.")
                except Exception as exc:
                    st.error(f"Capture failed: {exc}")

    if summary is not None and predictions is not None:
        st.divider()
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Traffic composition")
            st.bar_chart(pd.DataFrame({"Flows": [summary["normal"], summary["attack"]]}, index=["Normal", "Attack"]))
        with right:
            st.subheader("Attack types")
            breakdown = pd.Series(summary["breakdown"], name="Flows")
            st.bar_chart(breakdown) if not breakdown.empty else st.info("No attack types were predicted.")
        st.download_button("⬇️ Export CSV report", report_csv(summary, predictions), "quantum_ids_report.csv", "text/csv")
        try:
            st.download_button("⬇️ Export PDF report", report_pdf(summary), "quantum_ids_report.pdf", "application/pdf")
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")


if __name__ == "__main__":
    main()
