# ============================================================
# QUANTUM NETWORK INTRUSION DETECTION SYSTEM
# Hybrid Quantum Neural Network
#

# ============================================================

import os
import json
import threading
import logging
import platform
import traceback
import webbrowser
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn.functional as F

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# IMPORT YOUR HYBRID QNN
# ============================================================

from qnn_model import HybridQNN

# ============================================================
# IMPORT STAGE 2 -- ATTACK TYPE CLASSIFIER (classical model, runs after
# the Hybrid QNN flags a flow as ATTACK)
# ============================================================

try:
    from attack_classifier_model import AttackClassifier
    STAGE2_MODULE_AVAILABLE = True
except Exception:
    STAGE2_MODULE_AVAILABLE = False

# ============================================================
# OPTIONAL: SCAPY (for Live Capture tab)
# ============================================================

try:
    from scapy.all import conf

    # ROOT CAUSE FIX: recent Scapy versions default to a raw L3 socket
    # backend on Windows (L3WinSocket) instead of Npcap's real pcap
    # backend. That raw backend frequently binds to the wrong adapter
    # (or nothing at all) on newer Wi-Fi chipsets/drivers -- it never
    # raises an exception, it just never receives a packet. Forcing
    # conf.use_pcap = True BEFORE importing sniff/AsyncSniffer makes
    # Scapy use Npcap's actual packet driver, which is what reliably
    # sees traffic in promiscuous mode.
    conf.use_pcap = True

    from scapy.all import sniff, AsyncSniffer, IP, TCP, UDP, get_if_list
    if platform.system() == "Windows":
        from scapy.arch.windows import get_windows_if_list
    else:
        get_windows_if_list = None

    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False
    get_windows_if_list = None

# ============================================================
# PATHS / CONFIG
# ============================================================

MODEL_PATH = os.path.join("models", "hybrid_qnn.pth")
SCALER_PATH = os.path.join("processed", "scaler.pkl")

# Stage 2 -- attack type classifier (runs only when Stage 1 says ATTACK)
STAGE2_MODEL_PATH = os.path.join("models", "attack_classifier.pth")
STAGE2_SCALER_PATH = os.path.join("processed", "stage2_scaler.pkl")
STAGE2_CLASSES_PATH = os.path.join("models", "attack_classes.json")

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
ALERTS_FILE = "alerts.json"
REMEMBER_FILE = ".remember.json"
REPORTS_DIR = "reports"

# Recent Alerts / Top Attacks Today / Threat Timeline
MAX_ALERTS_STORED = 300
MAX_ALERTS_SHOWN = 8
DASHBOARD_TIMELINE_POINTS = 10

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "app.log")

# Order matters -- must match the column order used in preprocessing.py
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

# Rows above this trigger a "full dataset vs quick sample" prompt,
# since the quantum layer is simulated and can be slow on huge files.
LARGE_DATASET_THRESHOLD = 150000
QUICK_SAMPLE_SIZE = 40000
BATCH_SIZE = 2000

# Live capture tuning
EVAL_INTERVAL_MS = 3000
MIN_PACKETS_FOR_EVAL = 3
MAX_FLOW_AGE_SEC = 15
MAX_LIVE_ROWS = 200

# Heuristic thresholds (rough, for labeling only -- not model output)
DDOS_BPS_THRESHOLD = 50000
BRUTEFORCE_PORTS = {21, 22, 23, 3389, 3306, 5900}
BRUTEFORCE_PACKET_THRESHOLD = 8
SCAN_MIN_PPS = 5
SCAN_MAX_AVG_LEN = 80

# ============================================================
# APPLICATION COLORS  (kept identical to the original design)
# ============================================================

BG = "#F6F3FF"
HEADER = "#CDBBFF"
CARD = "#FFFFFF"
PRIMARY = "#8B6CFF"
PRIMARY_DARK = "#7356F3"
SUCCESS = "#6FCF97"
WARNING = "#F2C94C"
DANGER = "#EB5757"
TEXT = "#2D2D2D"
SUBTEXT = "#7A7A7A"
BORDER = "#E2D9FF"
NAV_ACTIVE = "#EDE6FF"

# Severity colors (reuses the existing palette -- no new colors introduced)
SEVERITY_COLORS = {
    "Critical": DANGER,
    "High": DANGER,
    "Medium": WARNING,
    "Low": SUCCESS,
    "Unknown": SUBTEXT,
}

# Threat Level Indicator badge colors/icons (reuses the existing palette --
# HIGH and CRITICAL share DANGER red like SEVERITY_COLORS already does;
# the icon is what visually separates them)
THREAT_BADGE_COLORS = {
    "LOW": SUCCESS,
    "MEDIUM": WARNING,
    "HIGH": DANGER,
    "CRITICAL": DANGER,
}

THREAT_BADGE_ICONS = {
    "LOW": "",
    "MEDIUM": "",
    "HIGH": "",
    "CRITICAL": "",
}

# ============================================================
# APPLICATION FONTS
# ============================================================

TITLE_FONT = ("Segoe UI", 24, "bold")
SUBTITLE_FONT = ("Segoe UI", 12)
HEADING_FONT = ("Segoe UI", 16, "bold")
TEXT_FONT = ("Segoe UI", 10)
BUTTON_FONT = ("Segoe UI", 10, "bold")
SMALL_FONT = ("Segoe UI", 9)
NAV_FONT = ("Segoe UI", 10, "bold")
STAT_FONT = ("Segoe UI", 22, "bold")
STAT_LABEL_FONT = ("Segoe UI", 9, "bold")

# ============================================================
# GLOBAL STATE
# ============================================================

state = {
    "user": None,
    "model": None,
    "model_ok": False,
    "model_error": "",
    "scaler": None,
    "scaler_ok": False,
    "scaler_error": "",
    "stage2_model": None,
    "stage2_scaler": None,
    "stage2_classes": [],
    "stage2_ok": False,
    "stage2_error": "",
    "df": None,
    "filename": None,
    "filepath": None,
    "results": None,
    "nav_buttons": {},
    "training_graph_image": None,
    "confusion_matrix_image": None,
    "capturing": False,
    "flow_store": {},
    "live_stats": {
        "total": 0, "attack": 0, "normal": 0, "attack_types": {}, "severity_counts": {},
        "packets_captured": 0, "ip_attacks": {},
    },
    "live_session_start": None,
}

# ============================================================
# USERS / AUTH
# ============================================================


def load_users():
    default_users = {
        "admin": {"name": "Administrator", "password": "admin123"}
    }

    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f, indent=4)
        return default_users

    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except Exception:
        users = {}

    # Make sure the default admin account always works, without
    # wiping out any accounts that already exist in users.json
    changed = False
    if "admin" not in users:
        users["admin"] = default_users["admin"]
        changed = True

    if changed:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

    return users


def check_login(username, password):
    users = load_users()
    user = users.get(username)
    if user and user.get("password") == password:
        return user
    return None


def save_new_user(username, name, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {"name": name, "password": password}
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
    return True, "Account created"


def load_remembered_user():
    if os.path.exists(REMEMBER_FILE):
        try:
            with open(REMEMBER_FILE, "r") as f:
                return json.load(f).get("username", "")
        except Exception:
            return ""
    return ""


def save_remembered_user(username, remember):
    try:
        if remember:
            with open(REMEMBER_FILE, "w") as f:
                json.dump({"username": username}, f)
        elif os.path.exists(REMEMBER_FILE):
            os.remove(REMEMBER_FILE)
    except Exception:
        pass


# ============================================================
# HISTORY
# ============================================================


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def clear_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)


# ============================================================
# ALERTS  (feeds Dashboard: Recent Alerts / Top Attacks Today / Threat Timeline)
# ============================================================


def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_alert(alert):
    """Append one alert (dict) to alerts.json, newest first, capped at
    MAX_ALERTS_STORED so the file never grows unbounded."""
    try:
        alerts = load_alerts()
        alerts.insert(0, alert)
        alerts = alerts[:MAX_ALERTS_STORED]
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=2)
    except Exception as e:
        log_exception("save_alert", e)


def make_alert(source, attack_type, severity, detail, confidence):
    now = datetime.now()
    return {
        "date": now.strftime("%d-%m-%Y"),
        "time": now.strftime("%H:%M:%S"),
        "source": source,
        "attack_type": attack_type or "Unknown",
        "severity": severity or "Unknown",
        "detail": detail,
        "confidence": confidence,
    }


# ============================================================
# SEVERITY / RECOMMENDATIONS  (rule-based, layered on top of Stage 1/2
# model output -- these do not change what the models predict, they
# just translate the prediction into a human-readable severity + advice)
# ============================================================

ATTACK_SEVERITY_KEYWORDS = [
    (("ddos", "dos", "flood"), "Critical"),
    (("botnet", "backdoor", "exploit", "trojan", "worm", "ransomware", "malware"), "Critical"),
    (("brute", "credential", "password"), "High"),
    (("infiltration", "injection", "web attack", "xss", "sql", "heartbleed"), "High"),
    (("scan", "recon", "probe", "fuzz"), "Medium"),
]

GENERIC_RECOMMENDATIONS = {
    "Critical": [
        "Isolate the affected host(s) from the network immediately.",
        "Escalate to the security team / incident response process.",
        "Preserve flow logs for forensic analysis before they rotate out.",
    ],
    "High": [
        "Review firewall and access-control rules for the affected hosts.",
        "Force credential resets for accounts on the targeted services.",
        "Increase monitoring on the affected segment for 24-48 hours.",
    ],
    "Medium": [
        "Confirm the flagged traffic isn't a known scan or maintenance job.",
        "Tighten rate-limiting on exposed/scanned services.",
    ],
    "Low": [
        "No immediate action required -- continue routine monitoring.",
    ],
    "Unknown": [
        "Manually review the flagged flows -- automatic attack typing was unavailable.",
    ],
}

ATTACK_TYPE_RECOMMENDATIONS = [
    (("ddos", "dos", "flood"), "Enable rate limiting / DDoS mitigation on internet-facing endpoints."),
    (("brute", "credential", "password"), "Enforce account lockout policies and multi-factor authentication."),
    (("scan", "recon", "probe"), "Restrict exposed ports with a firewall and disable unused services."),
    (("botnet", "malware", "trojan", "worm"), "Run endpoint malware scans on the source hosts and block outbound traffic."),
    (("injection", "web attack", "xss", "sql"), "Patch and validate input handling on the targeted web application."),
    (("backdoor", "exploit"), "Re-image or thoroughly audit the affected host and rotate all credentials."),
    (("infiltration",), "Audit internal network segmentation and watch for lateral movement."),
]


def severity_for_attack_type(attack_type):
    """Rough, rule-based severity label for a single Stage 2 attack-type
    string -- NOT a model output, just a readability layer on top of it."""
    if not attack_type or attack_type in ("-", "Unknown"):
        return "Unknown"
    low = attack_type.lower()
    for keywords, sev in ATTACK_SEVERITY_KEYWORDS:
        if any(k in low for k in keywords):
            return sev
    return "Medium"


def severity_color(severity):
    return SEVERITY_COLORS.get(severity, SUBTEXT)


def overall_severity(breakdown, threat_level=None):
    """Highest severity across an attack-type breakdown dict. Falls back
    to a threat-level based severity if there's no breakdown (e.g. Stage 2
    unavailable, or a single live flow)."""
    order = {"Unknown": -1, "Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    best = "Unknown"
    for name in (breakdown or {}):
        sev = severity_for_attack_type(name)
        if order.get(sev, -1) > order.get(best, -1):
            best = sev
    if best == "Unknown" and threat_level:
        best = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}.get(threat_level, "Unknown")
    return best


def compute_threat_level_full(attack_count, highest_severity):
    """Threat Level Indicator (LOW/MEDIUM/HIGH/CRITICAL) for a badge --
    combines attack volume with the highest severity already computed by
    overall_severity()/severity_for_attack_type(). This is a display-layer
    calculation only; it does not change Stage 1/Stage 2 model output."""
    rank = {"Unknown": -1, "Low": 0, "Medium": 1, "High": 2, "Critical": 3}.get(highest_severity, -1)
    if attack_count <= 0:
        return "LOW"
    if rank >= 3 or attack_count >= 50:
        return "CRITICAL"
    if rank == 2 or attack_count >= 15:
        return "HIGH"
    if rank == 1 or attack_count >= 5:
        return "MEDIUM"
    return "LOW"


def recommendations_for(severity, breakdown=None):
    """Ordered, de-duplicated list of recommendation strings for the given
    severity plus any attack-type-specific advice found in breakdown."""
    recs = list(GENERIC_RECOMMENDATIONS.get(severity, GENERIC_RECOMMENDATIONS["Unknown"]))
    seen = set(recs)
    for name in (breakdown or {}):
        low = name.lower()
        for keywords, rec in ATTACK_TYPE_RECOMMENDATIONS:
            if any(k in low for k in keywords) and rec not in seen:
                recs.append(rec)
                seen.add(rec)
    return recs


def attack_summary_text(results):
    """One paragraph, human-readable summary of an Upload Analysis result
    -- used by the 'Attack Summary' card and the PDF/CSV exports."""
    total = results.get("total", 0)
    attack = results.get("attack", 0)
    attack_pct = results.get("attack_pct", 0)
    threat = results.get("threat", "LOW")
    severity = results.get("severity", "Unknown")
    top_type = results.get("top_attack_type")

    if attack == 0:
        return (
            f"Analyzed {total:,} network flows from '{results.get('dataset', 'this dataset')}' "
            f"and found no attack traffic. Overall threat level: {threat}."
        )

    sentence = (
        f"Analyzed {total:,} network flows from '{results.get('dataset', 'this dataset')}' and "
        f"detected {attack:,} attack flows ({attack_pct}% of traffic), rated {severity} severity "
        f"(threat level: {threat})."
    )
    if top_type:
        count = (results.get("attack_breakdown") or {}).get(top_type, 0)
        sentence += f" The most common attack type was '{top_type}' ({count:,} flows)."
    return sentence


# ============================================================
# LOGGING
# ============================================================
#
# All loading/runtime errors go here (full detail, with tracebacks) so
# a beginner can zip up logs/app.log when asking for help, without
# needing to reproduce the error with a console open. The GUI itself
# only ever shows the short, friendly version of a message.


def _init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("nids_app")
    if logger.handlers:  # avoid duplicate handlers if this module reloads
        return logger

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    return logger


log = _init_logger()


def log_exception(context, exc):
    """Log an exception with full traceback under a short context label,
    e.g. log_exception('Stage 1 model load', e)."""
    log.error("%s failed: %s\n%s", context, exc, traceback.format_exc())


# ============================================================
# SYSTEM DIAGNOSTICS
# ============================================================
#
# A plain-text snapshot of the environment + which model/scaler files
# are present. Surfaced from Settings -> "System Diagnostics" so a
# beginner can see at a glance why something isn't working, and can
# copy/paste it when asking for help.


def run_system_diagnostics():
    """Returns a dict describing the environment and which required
    files are present. Never raises."""
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    diagnostics = {
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "platform": platform.platform(),
        "stage1_model_file": os.path.isfile(MODEL_PATH),
        "stage1_scaler_file": os.path.isfile(SCALER_PATH),
        "stage2_model_file": os.path.isfile(STAGE2_MODEL_PATH),
        "stage2_scaler_file": os.path.isfile(STAGE2_SCALER_PATH),
        "stage2_classes_file": os.path.isfile(STAGE2_CLASSES_PATH),
        "stage2_module_available": STAGE2_MODULE_AVAILABLE,
        "scapy_available": SCAPY_AVAILABLE,
        "model_ok": state.get("model_ok", False),
        "scaler_ok": state.get("scaler_ok", False),
        "stage2_ok": state.get("stage2_ok", False),
    }

    log.info(
        "Diagnostics: python=%s torch=%s cuda=%s stage1_model=%s stage1_scaler=%s "
        "stage2_model=%s stage2_scaler=%s stage2_classes=%s scapy=%s",
        diagnostics["python_version"], diagnostics["torch_version"], diagnostics["cuda_available"],
        diagnostics["model_ok"], diagnostics["scaler_ok"], diagnostics["stage2_ok"],
        diagnostics["stage2_scaler_file"], diagnostics["stage2_classes_file"], diagnostics["scapy_available"],
    )
    return diagnostics


def format_diagnostics_report(diagnostics):
    """Beginner-friendly plain text version of run_system_diagnostics(),
    suitable for a messagebox or a copyable text widget."""
    def yn(flag):
        return "Yes" if flag else "No"

    lines = [
        "System Diagnostics",
        "-------------------",
        f"Python version:        {diagnostics['python_version']}",
        f"PyTorch version:       {diagnostics['torch_version']}",
        f"GPU (CUDA) available:  {yn(diagnostics['cuda_available'])}",
        f"Platform:              {diagnostics['platform']}",
        "",
        f"Stage 1 (Hybrid QNN) model loaded:      {yn(diagnostics['model_ok'])}",
        f"Stage 1 scaler loaded:                  {yn(diagnostics['scaler_ok'])}",
        f"Stage 2 (attack type) model loaded:     {yn(diagnostics['stage2_ok'])}",
        f"Live capture (Scapy/Npcap) available:   {yn(diagnostics['scapy_available'])}",
    ]

    missing = []
    if not diagnostics["stage1_model_file"]:
        missing.append(f"  - {MODEL_PATH}")
    if not diagnostics["stage1_scaler_file"]:
        missing.append(f"  - {SCALER_PATH}")
    if not diagnostics["stage2_model_file"]:
        missing.append(f"  - {STAGE2_MODEL_PATH}")
    if not diagnostics["stage2_scaler_file"]:
        missing.append(f"  - {STAGE2_SCALER_PATH}")
    if not diagnostics["stage2_classes_file"]:
        missing.append(f"  - {STAGE2_CLASSES_PATH}")

    if missing:
        lines.append("")
        lines.append("Missing files:")
        lines.extend(missing)

    lines.append("")
    lines.append(f"Full logs saved to: {os.path.abspath(LOG_PATH)}")
    return "\n".join(lines)


def show_diagnostics_dialog():
    """Settings -> 'Run Diagnostics' button. Beginner-friendly summary
    in a plain messagebox -- no console needed."""
    diagnostics = run_system_diagnostics()
    messagebox.showinfo("System Diagnostics", format_diagnostics_report(diagnostics))


# ============================================================
# MODEL / SCALER LOADING
# ============================================================
#
# Every loader below follows the same shape:
#   1. Try to load the artifact.
#   2. On failure, log the full technical detail to logs/app.log.
#   3. Store a short, non-technical explanation in state[...] so the
#      GUI can show it (Settings/startup notice) without a stack trace.
#   4. Degrade gracefully -- a missing/broken model never crashes the
#      app, it just disables the feature that depended on it.


def load_model():
    try:
        model = HybridQNN()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )
        model.eval()
        state["model"] = model
        state["model_ok"] = True
        state["model_error"] = ""
        log.info("Stage 1 model loaded OK from %s", MODEL_PATH)
    except FileNotFoundError:
        msg = (
            f"Couldn't find the detection model at '{MODEL_PATH}'.\n"
            "Make sure you've trained it (or copied the models/ folder "
            "from wherever it was trained) before running the app."
        )
        log.warning("Stage 1 model file missing: %s", MODEL_PATH)
        state["model"] = None
        state["model_ok"] = False
        state["model_error"] = msg
    except Exception as e:
        msg = (
            "The detection model failed to load. This usually means the "
            "model file is corrupted or was saved with a different model "
            "definition than the one currently in qnn_model.py.\n"
            f"See {LOG_PATH} for the technical details."
        )
        log_exception("Stage 1 model load", e)
        state["model"] = None
        state["model_ok"] = False
        state["model_error"] = msg


def load_scaler():
    try:
        state["scaler"] = joblib.load(SCALER_PATH)
        state["scaler_ok"] = True
        state["scaler_error"] = ""
        log.info("Stage 1 scaler loaded OK from %s", SCALER_PATH)
    except FileNotFoundError:
        msg = (
            f"Couldn't find the feature scaler at '{SCALER_PATH}'.\n"
            "It's produced alongside the model during training -- make "
            "sure the processed/ folder was copied along with models/."
        )
        log.warning("Stage 1 scaler file missing: %s", SCALER_PATH)
        state["scaler"] = None
        state["scaler_ok"] = False
        state["scaler_error"] = msg
    except Exception as e:
        msg = (
            "The feature scaler failed to load (it may be from an "
            "incompatible scikit-learn version, or the file is corrupted).\n"
            f"See {LOG_PATH} for the technical details."
        )
        log_exception("Stage 1 scaler load", e)
        state["scaler"] = None
        state["scaler_ok"] = False
        state["scaler_error"] = msg


def load_attack_classifier():
    """Stage 2: loads the attack-type model, its own scaler, and the
    ordered class-name list saved by train_attack_classifier.py. If any
    of the three artifacts are missing, Stage 2 is simply disabled --
    Stage 1 (the Hybrid QNN) keeps working exactly as before, and the GUI
    falls back to showing 'Unknown' as the attack type."""
    if not STAGE2_MODULE_AVAILABLE:
        msg = (
            "The Stage 2 attack-type classifier module couldn't be "
            "imported, so the app will show 'Unknown' for attack type. "
            "Detection of ATTACK vs Normal traffic (Stage 1) is unaffected."
        )
        log.warning("Stage 2 disabled: attack_classifier_model import failed")
        state["stage2_ok"] = False
        state["stage2_error"] = msg
        return
    try:
        with open(STAGE2_CLASSES_PATH, "r") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError("attack_classes.json did not contain a non-empty list")

        model = AttackClassifier(input_dim=len(FEATURE_COLUMNS), num_classes=len(classes))
        model.load_state_dict(
            torch.load(STAGE2_MODEL_PATH, map_location=torch.device("cpu"))
        )
        model.eval()

        scaler = joblib.load(STAGE2_SCALER_PATH)

        state["stage2_model"] = model
        state["stage2_scaler"] = scaler
        state["stage2_classes"] = classes
        state["stage2_ok"] = True
        state["stage2_error"] = ""
        log.info("Stage 2 model loaded OK (%d attack types) from %s", len(classes), STAGE2_MODEL_PATH)
    except FileNotFoundError as e:
        msg = (
            "The attack-type classifier hasn't been trained yet, so the "
            "app can still flag ATTACK vs Normal traffic, but can't name "
            "the specific attack type.\n\n"
            "To enable it, run train_attack_classifier.py, then restart "
            "the app."
        )
        log.warning("Stage 2 file missing: %s", e)
        state["stage2_model"] = None
        state["stage2_scaler"] = None
        state["stage2_classes"] = []
        state["stage2_ok"] = False
        state["stage2_error"] = msg
    except RuntimeError as e:
        # Most common cause: attack_classes.json and attack_classifier.pth
        # are out of sync (retrained with a different class count but only
        # one of the two files was updated/copied).
        msg = (
            "The attack-type model doesn't match its class list (this "
            "usually happens if training was interrupted, or only one of "
            "the Stage 2 output files was updated).\n\n"
            "Re-run train_attack_classifier.py so the model and class "
            "list are regenerated together."
        )
        log_exception("Stage 2 model/class mismatch", e)
        state["stage2_model"] = None
        state["stage2_scaler"] = None
        state["stage2_classes"] = []
        state["stage2_ok"] = False
        state["stage2_error"] = msg
    except Exception as e:
        msg = (
            "The attack-type classifier failed to load, so the app will "
            "show 'Unknown' for attack type. Detection of ATTACK vs Normal "
            "traffic (Stage 1) is unaffected.\n"
            f"See {LOG_PATH} for the technical details."
        )
        log_exception("Stage 2 (attack classifier) load", e)
        state["stage2_model"] = None
        state["stage2_scaler"] = None
        state["stage2_classes"] = []
        state["stage2_ok"] = False
        state["stage2_error"] = msg


def classify_attack_type(features_dict):
    """Stage 2, single flow (used by Live Capture). Takes the same
    11-feature dict already built for Stage 1 and returns
    (attack_type_label, confidence_pct). Returns ('Unknown', 0.0) if
    Stage 2 isn't loaded or prediction fails for any reason -- this
    must never break Stage 1's Normal/Attack result."""
    if not state["stage2_ok"]:
        return "Unknown", 0.0
    try:
        row = np.array([[features_dict[c] for c in FEATURE_COLUMNS]])
        scaled = state["stage2_scaler"].transform(row)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            output = state["stage2_model"](tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
        idx = int(pred.item())
        classes = state["stage2_classes"]
        label = classes[idx] if 0 <= idx < len(classes) else "Unknown"
        return label, round(float(conf.item()) * 100, 1)
    except Exception:
        return "Unknown", 0.0


def classify_attack_types_batch(x_array):
    """Stage 2, batch (used by Upload Analysis). x_array is a 2D numpy
    array of raw (unscaled) 11-feature rows for flows Stage 1 already
    flagged ATTACK. Returns a list of attack_type strings, same length
    and order as x_array. Never raises -- falls back to 'Unknown'."""
    n = len(x_array)
    if not state["stage2_ok"] or n == 0:
        return ["Unknown"] * n
    try:
        scaled = state["stage2_scaler"].transform(x_array)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        classes = state["stage2_classes"]
        labels = []
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                batch = tensor[start:start + BATCH_SIZE]
                output = state["stage2_model"](batch)
                probs = F.softmax(output, dim=1)
                _, preds = torch.max(probs, dim=1)
                for idx in preds.numpy().tolist():
                    labels.append(classes[idx] if 0 <= idx < len(classes) else "Unknown")
        return labels
    except Exception:
        return ["Unknown"] * n


# ============================================================
# HELPERS
# ============================================================


def format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def threat_level_from_pct(attack_pct):
    if attack_pct < 10:
        return "LOW"
    elif attack_pct < 30:
        return "MEDIUM"
    else:
        return "HIGH"


def rounded_button(parent, text, command, bg=PRIMARY, fg=TEXT, width=22, height=2, font=BUTTON_FONT):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=bg,
        activeforeground=TEXT,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
        width=width,
        height=height,
        cursor="hand2",
        font=font,
    )


def card_frame(parent, **kwargs):
    defaults = dict(bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    defaults.update(kwargs)
    return tk.Frame(parent, **defaults)


def render_threat_badge(parent, level, bg=CARD):
    """Threat Level Indicator: a colored LOW/MEDIUM/HIGH/CRITICAL badge,
    styled the same simple colored-label way as status_pill()/mini_stat()."""
    color = THREAT_BADGE_COLORS.get(level, SUBTEXT)
    icon = THREAT_BADGE_ICONS.get(level, "")
    tk.Label(
        parent, text=f"{icon}{' ' if icon else ''}{level}", bg=bg, fg=color,
        font=("Segoe UI", 16, "bold"),
    ).pack(pady=6)


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()
root.title("Quantum Network Intrusion Detection System")
root.geometry("1400x850")
root.minsize(1100, 700)
root.configure(bg=BG)
root.resizable(True, True)

main_frame = tk.Frame(root, bg=BG)
main_frame.pack(fill="both", expand=True)

content_frame = None  # set once the app shell is built


def clear_screen():
    for widget in main_frame.winfo_children():
        widget.destroy()


def clear_content():
    if content_frame is not None:
        for widget in content_frame.winfo_children():
            widget.destroy()


# ============================================================
# LOGIN SCREEN
# ============================================================


def login_screen():
    clear_screen()

    header = tk.Frame(main_frame, bg=HEADER, height=150)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header,
        text="QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
        bg=HEADER,
        fg=TEXT,
        font=TITLE_FONT,
    ).pack(pady=(28, 5))

    tk.Label(
        header,
        text="Hybrid Quantum Neural Network Based Intrusion Detection",
        bg=HEADER,
        fg=TEXT,
        font=SUBTITLE_FONT,
    ).pack()

    outer = tk.Frame(main_frame, bg=BG)
    outer.pack(expand=True)

    card = card_frame(outer)
    card.pack(pady=60, ipadx=50, ipady=35)
    card.columnconfigure(0, weight=1)

    FIELD_WIDTH = 30

    tk.Label(card, text="Login", bg=CARD, fg=TEXT, font=("Segoe UI", 22, "bold")).grid(
        row=0, column=0, pady=(5, 25)
    )

    tk.Label(card, text="Username", bg=CARD, fg=SUBTEXT, font=TEXT_FONT).grid(
        row=1, column=0, pady=(0, 4)
    )
    username_var = tk.StringVar(value=load_remembered_user())
    username_entry = tk.Entry(
        card, textvariable=username_var, font=TEXT_FONT, width=FIELD_WIDTH, justify="center",
        relief="flat", highlightthickness=1, highlightbackground=BORDER, highlightcolor=PRIMARY,
    )
    username_entry.grid(row=2, column=0, ipady=6, pady=(0, 18))

    tk.Label(card, text="Password", bg=CARD, fg=SUBTEXT, font=TEXT_FONT).grid(
        row=3, column=0, pady=(0, 4)
    )
    password_var = tk.StringVar()
    password_entry = tk.Entry(
        card, textvariable=password_var, show="*", font=TEXT_FONT, width=FIELD_WIDTH, justify="center",
        relief="flat", highlightthickness=1, highlightbackground=BORDER, highlightcolor=PRIMARY,
    )
    password_entry.grid(row=4, column=0, ipady=6, pady=(0, 18))

    remember_var = tk.BooleanVar(value=bool(load_remembered_user()))
    tk.Checkbutton(
        card, text="Remember Me", variable=remember_var, bg=CARD, fg=SUBTEXT,
        font=SMALL_FONT, activebackground=CARD, selectcolor=CARD,
    ).grid(row=5, column=0, pady=(0, 18))

    error_label = tk.Label(card, text="", bg=CARD, fg=DANGER, font=SMALL_FONT)
    error_label.grid(row=6, column=0, pady=(0, 4))

    def do_login(event=None):
        username = username_var.get().strip()
        password = password_var.get().strip()

        user = check_login(username, password)
        if user:
            state["user"] = {"username": username, "name": user.get("name", username)}
            save_remembered_user(username, remember_var.get())
            build_app_shell()
        else:
            error_label.config(text="Invalid username or password")

    tk.Button(
        card, text="LOGIN", command=do_login, bg=PRIMARY, fg=TEXT,
        activebackground=PRIMARY, activeforeground=TEXT, relief="flat",
        bd=0, highlightthickness=1, highlightbackground=BORDER,
        width=FIELD_WIDTH, height=2, cursor="hand2", font=BUTTON_FONT,
    ).grid(row=7, column=0, pady=(4, 18))

    root.bind("<Return>", do_login)

    tk.Label(
        card, text="Default   Username: admin    Password: admin123",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
    ).grid(row=8, column=0, pady=(0, 12))

    signup_link = tk.Label(
        card, text="Don't have an account?  Sign Up", bg=CARD, fg=PRIMARY_DARK,
        font=SMALL_FONT, cursor="hand2",
    )
    signup_link.grid(row=9, column=0, pady=(0, 0))
    signup_link.bind("<Button-1>", lambda e: signup_screen())

    footer = tk.Frame(main_frame, bg=BG)
    footer.pack(side="bottom", pady=20)
    tk.Label(footer, text="Network security analysis dashboard", bg=BG, fg=SUBTEXT, font=SMALL_FONT).pack()

    username_entry.focus_set()


# ============================================================
# SIGN UP SCREEN
# ============================================================


def signup_screen():
    clear_screen()
    root.unbind("<Return>")

    header = tk.Frame(main_frame, bg=HEADER, height=150)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header, text="QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
        bg=HEADER, fg=TEXT, font=TITLE_FONT,
    ).pack(pady=(28, 5))
    tk.Label(
        header, text="Create a new account", bg=HEADER, fg=TEXT, font=SUBTITLE_FONT
    ).pack()

    outer = tk.Frame(main_frame, bg=BG)
    outer.pack(expand=True)

    card = card_frame(outer)
    card.pack(pady=50, ipadx=50, ipady=30)

    tk.Label(card, text="Sign Up", bg=CARD, fg=TEXT, font=("Segoe UI", 22, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(5, 20)
    )

    fields = {}

    def field_row(row, label, show=None):
        tk.Label(card, text=label, bg=CARD, fg=SUBTEXT, font=TEXT_FONT).grid(
            row=row, column=0, sticky="w", pady=(0, 4)
        )
        var = tk.StringVar()
        entry = tk.Entry(
            card, textvariable=var, show=show, font=TEXT_FONT, width=30,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, highlightcolor=PRIMARY,
        )
        entry.grid(row=row + 1, column=0, columnspan=2, ipady=6, pady=(0, 12))
        return var

    name_var = field_row(1, "Full Name")
    username_var = field_row(3, "Username")
    password_var = field_row(5, "Password", show="*")
    confirm_var = field_row(7, "Confirm Password", show="*")

    error_label = tk.Label(card, text="", bg=CARD, fg=DANGER, font=SMALL_FONT)
    error_label.grid(row=9, column=0, columnspan=2)

    def do_signup():
        name = name_var.get().strip()
        username = username_var.get().strip()
        password = password_var.get().strip()
        confirm = confirm_var.get().strip()

        if not name or not username or not password:
            error_label.config(text="Please fill in all fields")
            return
        if password != confirm:
            error_label.config(text="Passwords do not match")
            return
        if len(password) < 4:
            error_label.config(text="Password must be at least 4 characters")
            return

        ok, message = save_new_user(username, name, password)
        if not ok:
            error_label.config(text=message)
            return

        messagebox.showinfo("Account Created", "Your account has been created. Please log in.")
        login_screen()

    tk.Button(
        card, text="CREATE ACCOUNT", command=do_signup, bg=SUCCESS, fg=TEXT,
        activebackground=SUCCESS, activeforeground=TEXT, relief="flat",
        bd=0, highlightthickness=1, highlightbackground=BORDER,
        width=28, height=2, cursor="hand2", font=BUTTON_FONT,
    ).grid(row=10, column=0, columnspan=2, pady=(5, 5))

    back_link = tk.Label(
        card, text="Already have an account?  Login", bg=CARD, fg=PRIMARY_DARK,
        font=SMALL_FONT, cursor="hand2",
    )
    back_link.grid(row=11, column=0, columnspan=2, pady=(12, 0))
    back_link.bind("<Button-1>", lambda e: login_screen())


# ============================================================
# APP SHELL  (header + nav bar + content area)
# ============================================================


def build_app_shell():
    global content_frame

    root.unbind("<Return>")
    clear_screen()

    # -------------------- HEADER --------------------
    header = tk.Frame(main_frame, bg=HEADER, height=95)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header, text="QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
        bg=HEADER, fg=TEXT, font=("Segoe UI", 19, "bold"),
    ).pack(side="left", padx=25, pady=(18, 0))

    user_box = tk.Frame(header, bg=HEADER)
    user_box.pack(side="right", padx=25, pady=(18, 0))
    tk.Label(
        user_box, text=f"{state['user']['name']}", bg=HEADER, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="e")
    tk.Label(
        user_box, text="Hybrid Quantum Neural Network", bg=HEADER, fg=SUBTEXT, font=SMALL_FONT
    ).pack(anchor="e")

    # -------------------- NAV BAR --------------------
    nav_bar = tk.Frame(main_frame, bg=CARD, height=52, highlightbackground=BORDER, highlightthickness=1)
    nav_bar.pack(fill="x")
    nav_bar.pack_propagate(False)

    nav_items = [
        ("dashboard", "Dashboard", view_dashboard),
        ("analyze", "Upload & Analyze", view_analyze),
        ("live", "Live Capture", view_live_capture),
        ("training", "Training Results", view_training_results),
        ("history", "Analysis History", view_history),
        ("settings", "Settings", view_settings),
        ("about", "About", view_about),
    ]

    state["nav_buttons"] = {}

    nav_left = tk.Frame(nav_bar, bg=CARD)
    nav_left.pack(side="left", padx=15)

    for key, label, handler in nav_items:
        btn = tk.Button(
            nav_left, text=label, command=lambda h=handler, k=key: switch_view(k, h),
            bg=CARD, fg=TEXT, relief="flat", bd=0, cursor="hand2",
            font=NAV_FONT, padx=16, pady=10,
        )
        btn.pack(side="left")
        state["nav_buttons"][key] = btn

    tk.Button(
        nav_bar, text="Logout", command=logout, bg=CARD, fg=DANGER,
        relief="flat", bd=0, cursor="hand2", font=NAV_FONT, padx=16, pady=10,
    ).pack(side="right", padx=15)

    # -------------------- CONTENT AREA --------------------
    content_frame = tk.Frame(main_frame, bg=BG)
    content_frame.pack(fill="both", expand=True)

    if not state["model_ok"]:
        load_model()
    if not state["scaler_ok"]:
        load_scaler()

    switch_view("dashboard", view_dashboard)


def switch_view(key, handler):
    for k, btn in state["nav_buttons"].items():
        if k == key:
            btn.config(bg=NAV_ACTIVE, fg=PRIMARY_DARK)
        else:
            btn.config(bg=CARD, fg=TEXT)
    clear_content()
    handler()


def logout():
    state["capturing"] = False
    state["user"] = None
    state["df"] = None
    state["filename"] = None
    state["filepath"] = None
    state["results"] = None
    login_screen()


# ============================================================
# VIEW: DASHBOARD
# ============================================================


def status_pill(parent, label, ok):
    dot = "OK" if ok else "NOT READY"
    text = f"{dot} {label}"
    tk.Label(
        parent, text=text, bg=CARD, fg=TEXT, font=BUTTON_FONT,
        padx=18, pady=12,
    ).pack(side="left", padx=10)


def view_dashboard():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=30, pady=25)

    tk.Label(
        wrap, text=f"Welcome back, {state['user']['name']}",
        bg=BG, fg=TEXT, font=HEADING_FONT,
    ).pack(anchor="w")

    tk.Label(
        wrap, text="Here's the current status of your intrusion detection system.",
        bg=BG, fg=SUBTEXT, font=TEXT_FONT,
    ).pack(anchor="w", pady=(0, 20))

    status_card = card_frame(wrap)
    status_card.pack(fill="x", pady=(0, 25))
    status_row = tk.Frame(status_card, bg=CARD)
    status_row.pack(pady=15)

    status_pill(status_row, "Hybrid QNN Ready", state["model_ok"])
    status_pill(status_row, "Dataset Ready", state["df"] is not None)
    status_pill(status_row, "System Ready", state["model_ok"] and state["scaler_ok"])

    if not state["model_ok"]:
        tk.Label(
            status_card,
            text=f"Warning: could not load {MODEL_PATH}. Train the model first using train_qnn.py.",
            bg=CARD, fg=DANGER, font=SMALL_FONT,
        ).pack(pady=(0, 10))

    if not state["scaler_ok"]:
        tk.Label(
            status_card,
            text=f"Warning: could not load {SCALER_PATH}. Run preprocessing.py first.",
            bg=CARD, fg=DANGER, font=SMALL_FONT,
        ).pack(pady=(0, 10))

    # Quick action cards
    quick_row = tk.Frame(wrap, bg=BG)
    quick_row.pack(fill="x", pady=(0, 25))

    action_card = card_frame(quick_row, width=350, height=140)
    action_card.pack(side="left", padx=(0, 20))
    action_card.pack_propagate(False)
    tk.Label(action_card, text="Analyze New Traffic", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(18, 5)
    )
    tk.Label(
        action_card, text="Upload a CSV of network flows and run\nthe Hybrid QNN prediction.",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="left",
    ).pack(anchor="w", padx=18)
    rounded_button(
        action_card, "Go to Upload", lambda: switch_view("analyze", view_analyze),
        width=18, height=1,
    ).pack(anchor="w", padx=18, pady=12)

    history = load_history()
    hist_card = card_frame(quick_row, width=350, height=140)
    hist_card.pack(side="left")
    hist_card.pack_propagate(False)
    tk.Label(hist_card, text="Recent Activity", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(18, 5)
    )
    if history:
        last = history[0]
        tk.Label(
            hist_card,
            text=f"Last scan: {last['dataset']}\n{last['date']}  -  Threat: {last['threat']}",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="left",
        ).pack(anchor="w", padx=18)
    else:
        tk.Label(
            hist_card, text="No scans yet. Run your first analysis!",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(anchor="w", padx=18)
    rounded_button(
        hist_card, "View History", lambda: switch_view("history", view_history),
        bg=SUCCESS, width=18, height=1,
    ).pack(anchor="w", padx=18, pady=12)

    # Model info card
    info_card = card_frame(wrap)
    info_card.pack(fill="x")
    tk.Label(info_card, text="Model Information", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 10)
    )
    info_grid = tk.Frame(info_card, bg=CARD)
    info_grid.pack(fill="x", padx=20, pady=(0, 20))

    info_items = [
        ("Model", "Hybrid Quantum Neural Network"),
        ("Framework", "PyTorch + PennyLane"),
        ("Datasets", "NF-UQ-NIDS-v2, LycoS-IDS, CIC-DDoS2019"),
        ("Input Features", "11"),
    ]
    for i, (k, v) in enumerate(info_items):
        col = tk.Frame(info_grid, bg=CARD)
        col.grid(row=0, column=i, sticky="w", padx=(0, 40))
        tk.Label(col, text=k, bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack(anchor="w")
        tk.Label(col, text=v, bg=CARD, fg=PRIMARY_DARK, font=TEXT_FONT).pack(anchor="w")

    # ---- Threat Timeline ----
    timeline_card = card_frame(wrap)
    timeline_card.pack(fill="x", pady=(25, 0))
    tk.Label(timeline_card, text="Threat Timeline", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 5)
    )
    recent_scans = list(reversed(history[:DASHBOARD_TIMELINE_POINTS]))
    if recent_scans:
        draw_threat_timeline_chart(timeline_card, recent_scans)
    else:
        tk.Label(
            timeline_card, text="No scans yet -- run an analysis to start building the timeline.",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(anchor="w", padx=20, pady=(0, 20))

    # ---- Top Attacks Today + Recent Alerts ----
    alerts = load_alerts()
    today_str = datetime.now().strftime("%d-%m-%Y")

    bottom_row = tk.Frame(wrap, bg=BG)
    bottom_row.pack(fill="x", pady=(20, 0))

    top_card = card_frame(bottom_row, width=440, height=260)
    top_card.pack(side="left", padx=(0, 20), fill="y")
    top_card.pack_propagate(False)
    tk.Label(top_card, text="Top Attacks Today", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(15, 8)
    )

    today_alerts = [a for a in alerts if a.get("date") == today_str]
    today_counts = {}
    for a in today_alerts:
        name = a.get("attack_type", "Unknown")
        today_counts[name] = today_counts.get(name, 0) + 1

    if today_counts:
        ranked_today = sorted(today_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for name, count in ranked_today:
            sev = severity_for_attack_type(name)
            row = tk.Frame(top_card, bg=CARD)
            row.pack(fill="x", padx=18, pady=3)
            tk.Label(row, text=name, bg=CARD, fg=TEXT, font=SMALL_FONT, anchor="w").pack(side="left")
            tk.Label(
                row, text=f"{count}  -  {sev}", bg=CARD, fg=severity_color(sev), font=SMALL_FONT
            ).pack(side="right")
    else:
        tk.Label(
            top_card, text="No attacks recorded today.", bg=CARD, fg=SUBTEXT, font=SMALL_FONT
        ).pack(anchor="w", padx=18)

    alerts_card = card_frame(bottom_row, height=260)
    alerts_card.pack(side="left", fill="both", expand=True)
    alerts_card.pack_propagate(False)
    tk.Label(alerts_card, text="Recent Alerts", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(15, 8)
    )

    if alerts:
        for a in alerts[:MAX_ALERTS_SHOWN]:
            sev = a.get("severity", "Unknown")
            row = tk.Frame(alerts_card, bg=CARD)
            row.pack(fill="x", padx=18, pady=3)
            tk.Label(
                row, text=f"{a.get('date', '')} {a.get('time', '')}", bg=CARD, fg=SUBTEXT,
                font=SMALL_FONT, width=16, anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=f"[{a.get('source', '-')}] {a.get('attack_type', 'Unknown')}",
                bg=CARD, fg=TEXT, font=SMALL_FONT, anchor="w",
            ).pack(side="left", padx=(6, 6))
            tk.Label(row, text=sev, bg=CARD, fg=severity_color(sev), font=SMALL_FONT).pack(side="right")
    else:
        tk.Label(
            alerts_card, text="No alerts yet. Attacks from Upload Analysis and Live Capture will appear here.",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="left", wraplength=380,
        ).pack(anchor="w", padx=18)


def draw_threat_timeline_chart(parent, scans):
    """Line chart of Attack % across the last few scans (History), oldest
    to newest, color-coded by threat level at each point."""
    labels = [s.get("date", "")[-5:] for s in scans]  # trim to HH:MM for readability
    values = [s.get("attack_pct", 0) for s in scans]
    colors = [threat_color(s.get("threat", "LOW")) for s in scans]

    fig = Figure(figsize=(10.5, 2.6), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)

    ax.plot(labels, values, color=PRIMARY, linewidth=2, zorder=1)
    ax.scatter(labels, values, color=colors, s=45, zorder=2)

    ax.set_ylabel("Attack %", fontsize=9, color=SUBTEXT)
    ax.tick_params(axis="both", labelsize=8, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=BORDER, linewidth=0.6)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=15, pady=(0, 15), fill="x")


# ============================================================
# VIEW: UPLOAD & ANALYZE
# ============================================================

analyze_widgets = {}


def view_analyze():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=25, pady=20)

    # ---------------- LEFT: Upload ----------------
    left = card_frame(wrap, width=340)
    left.pack(side="left", fill="y", padx=(0, 20))
    left.pack_propagate(False)

    tk.Label(left, text="Upload Network Traffic", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        pady=(20, 15)
    )

    rounded_button(left, "Browse CSV File", browse_file, width=24).pack(pady=5)

    file_info = tk.Frame(left, bg=CARD)
    file_info.pack(pady=15, fill="x", padx=20)
    analyze_widgets["file_info"] = file_info

    render_file_info()

    analyze_widgets["analyze_btn"] = rounded_button(
        left, "Analyze Dataset", start_analysis, bg=SUCCESS, width=24
    )
    analyze_widgets["analyze_btn"].pack(pady=15)
    analyze_widgets["analyze_btn"].config(state="normal" if state["df"] is not None else "disabled")

    tk.Label(left, text="Status", bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack(pady=(20, 2))
    analyze_widgets["status_label"] = tk.Label(
        left, text="Waiting for file...", bg=CARD, fg=SUBTEXT, font=SMALL_FONT
    )
    analyze_widgets["status_label"].pack()

    analyze_widgets["progress"] = ttk.Progressbar(left, length=260, mode="determinate")
    analyze_widgets["progress"].pack(pady=15)

    # ---------------- RIGHT: Scrollable Report ----------------
    right_outer = tk.Frame(wrap, bg=BG)
    right_outer.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(right_outer, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=canvas.yview)
    report = tk.Frame(canvas, bg=BG)

    report.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=report, anchor="nw", width=1080)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    analyze_widgets["report"] = report
    render_report()


def render_file_info():
    for w in analyze_widgets["file_info"].winfo_children():
        w.destroy()

    if state["filename"] is None:
        tk.Label(
            analyze_widgets["file_info"], text="No file selected", bg=CARD, fg=SUBTEXT, font=SMALL_FONT
        ).pack()
        return

    rows = [
        ("File Name", state["filename"]),
        ("Size", format_size(os.path.getsize(state["filepath"]))),
        ("Rows", f"{len(state['df']):,}"),
    ]
    for label, value in rows:
        row = tk.Frame(analyze_widgets["file_info"], bg=CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT, width=10, anchor="w").pack(side="left")
        tk.Label(row, text=value, bg=CARD, fg=TEXT, font=SMALL_FONT, anchor="w").pack(side="left")


def browse_file():
    path = filedialog.askopenfilename(
        title="Select Network Traffic CSV", filetypes=[("CSV Files", "*.csv")]
    )
    if not path:
        return

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read CSV file:\n{e}")
        return

    state["df"] = df
    state["filename"] = os.path.basename(path)
    state["filepath"] = path
    state["results"] = None

    render_file_info()
    analyze_widgets["analyze_btn"].config(state="normal")
    analyze_widgets["status_label"].config(text="File loaded successfully", fg=SUCCESS)
    render_report()


def start_analysis():
    if state["df"] is None:
        return
    if not (state["model_ok"] and state["scaler_ok"]):
        messagebox.showerror(
            "Model Not Ready",
            "The Hybrid QNN model or scaler could not be loaded.\n"
            "Make sure models/hybrid_qnn.pth and processed/scaler.pkl exist.",
        )
        return

    df = state["df"]
    sample_only = False

    if len(df) > LARGE_DATASET_THRESHOLD:
        answer = messagebox.askyesno(
            "Large Dataset",
            f"This file has {len(df):,} flows. Analyzing the full dataset with the "
            f"quantum layer can take a while.\n\n"
            f"Analyze a quick random sample of {QUICK_SAMPLE_SIZE:,} flows instead?\n"
            f"(Choose 'No' to analyze all flows.)",
        )
        sample_only = answer

    analyze_widgets["analyze_btn"].config(state="disabled")
    analyze_widgets["status_label"].config(text="Analyzing...", fg=PRIMARY_DARK)
    analyze_widgets["progress"]["value"] = 0

    thread = threading.Thread(target=run_analysis, args=(sample_only,), daemon=True)
    thread.start()


def run_analysis(sample_only):
    try:
        df = state["df"].copy()

        # Normalize column names the same way combine_datasets.py does
        df.columns = df.columns.str.strip().str.lower()

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            root.after(
                0, lambda: analysis_failed(f"Missing required columns: {', '.join(missing)}")
            )
            return

        if sample_only and len(df) > QUICK_SAMPLE_SIZE:
            df = df.sample(n=QUICK_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        X = df[FEATURE_COLUMNS].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)

        total_flows = len(X)
        if total_flows == 0:
            root.after(0, lambda: analysis_failed("No valid flows found after cleaning the data."))
            return

        X_scaled = state["scaler"].transform(X.values)

        model = state["model"]
        predictions = []
        confidences = []

        with torch.no_grad():
            for start in range(0, total_flows, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total_flows)
                batch = torch.tensor(X_scaled[start:end], dtype=torch.float32)

                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, dim=1)

                predictions.extend(preds.numpy().tolist())
                confidences.extend(conf.numpy().tolist())

                progress_pct = (end / total_flows) * 100
                root.after(0, lambda p=progress_pct, e=end: update_progress(p, e, total_flows))

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        attack_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())
        attack_pct = round((attack_count / total_flows) * 100, 2)
        avg_confidence = round(float(confidences.mean()) * 100, 2)
        threat_level = threat_level_from_pct(attack_pct)
        status = "ATTACK DETECTED" if attack_count > 0 and attack_pct >= 1 else "NORMAL TRAFFIC"

        # ---- Stage 2: classify the type of each flow Stage 1 flagged ----
        attack_breakdown = {}
        top_attack_type = None
        if attack_count > 0:
            attack_indices = np.where(predictions == 1)[0]
            attack_types = classify_attack_types_batch(X.values[attack_indices])
            for label in attack_types:
                attack_breakdown[label] = attack_breakdown.get(label, 0) + 1
            if attack_breakdown:
                top_attack_type = max(attack_breakdown, key=attack_breakdown.get)

        severity = overall_severity(attack_breakdown, threat_level) if attack_count > 0 else "Low"
        recommendations = recommendations_for(severity, attack_breakdown)

        results = {
            "dataset": state["filename"],
            "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "total": total_flows,
            "normal": normal_count,
            "attack": attack_count,
            "attack_pct": attack_pct,
            "threat": threat_level,
            "confidence": avg_confidence,
            "status": status,
            "sampled": sample_only,
            "attack_breakdown": attack_breakdown,
            "top_attack_type": top_attack_type,
            "stage2_ok": state["stage2_ok"],
            "severity": severity,
            "recommendations": recommendations,
        }

        state["results"] = results
        save_history_entry(results)

        if attack_count > 0:
            save_alert(make_alert(
                source="Upload Analysis",
                attack_type=top_attack_type or "Unknown",
                severity=severity,
                detail=f"{state['filename']} -- {attack_count:,} attack flow(s) ({attack_pct}%)",
                confidence=avg_confidence,
            ))

        root.after(0, analysis_complete)

    except Exception as e:
        root.after(0, lambda: analysis_failed(str(e)))


def update_progress(pct, done, total):
    analyze_widgets["progress"]["value"] = pct
    analyze_widgets["status_label"].config(text=f"Analyzing... {done:,}/{total:,} flows", fg=PRIMARY_DARK)


def analysis_complete():
    analyze_widgets["analyze_btn"].config(state="normal")
    analyze_widgets["status_label"].config(text="Analysis complete", fg=SUCCESS)
    analyze_widgets["progress"]["value"] = 100
    render_report()


def analysis_failed(message):
    analyze_widgets["analyze_btn"].config(state="normal")
    analyze_widgets["status_label"].config(text="Analysis failed", fg=DANGER)
    analyze_widgets["progress"]["value"] = 0
    messagebox.showerror("Analysis Failed", message)


def render_report():
    report = analyze_widgets["report"]
    for w in report.winfo_children():
        w.destroy()

    results = state["results"]

    if results is None:
        placeholder = card_frame(report)
        placeholder.pack(fill="x", pady=(0, 15))
        tk.Label(
            placeholder, text="Analysis Report", bg=CARD, fg=TEXT, font=BUTTON_FONT
        ).pack(anchor="w", padx=20, pady=(15, 5))
        tk.Label(
            placeholder,
            text="Upload a CSV file and click 'Analyze Dataset' to see the\nThreat Status, Pie Chart, Threat Meter and full report here.",
            bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 20))
        return

    is_attack = results["status"] == "ATTACK DETECTED"
    status_color = DANGER if is_attack else SUCCESS
    status_icon = "ATTACK" if is_attack else "NORMAL"

    # ---- Threat Status Card ----
    status_card = card_frame(report)
    status_card.pack(fill="x", pady=(0, 15))

    tk.Label(status_card, text="Threat Status", bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack(pady=(18, 2))
    tk.Label(
        status_card, text=f"{status_icon} {results['status']}", bg=CARD, fg=status_color,
        font=("Segoe UI", 20, "bold"),
    ).pack()

    sub_row = tk.Frame(status_card, bg=CARD)
    sub_row.pack(pady=(10, 18))

    def mini_stat(parent, label, value, color=TEXT):
        col = tk.Frame(parent, bg=CARD)
        col.pack(side="left", padx=25)
        tk.Label(col, text=value, bg=CARD, fg=color, font=("Segoe UI", 15, "bold")).pack()
        tk.Label(col, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack()

    mini_stat(sub_row, "Confidence", f"{results['confidence']}%", PRIMARY_DARK)
    mini_stat(sub_row, "Threat Level", results["threat"], threat_color(results["threat"]))
    if results.get("sampled"):
        mini_stat(sub_row, "Mode", "Sampled")

    # ---- Threat Level Indicator ----
    indicator_card = card_frame(report)
    indicator_card.pack(fill="x", pady=(0, 15))
    tk.Label(indicator_card, text="Threat Level Indicator", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 0)
    )
    indicator_level = compute_threat_level_full(results.get("attack", 0), results.get("severity", "Unknown"))
    render_threat_badge(indicator_card, indicator_level)
    tk.Label(
        indicator_card,
        text=f"Based on {results.get('attack', 0):,} detected attack flow(s), "
             f"highest severity: {results.get('severity', 'Unknown')}.",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
    ).pack(anchor="w", padx=20, pady=(0, 15))

    # ---- Pie chart + Threat meter side by side ----
    viz_row = tk.Frame(report, bg=BG)
    viz_row.pack(fill="x", pady=(0, 15))

    pie_card = card_frame(viz_row, width=520, height=340)
    pie_card.pack(side="left", padx=(0, 15))
    pie_card.pack_propagate(False)
    tk.Label(pie_card, text="Traffic Breakdown", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(pady=(15, 0))
    draw_pie_chart(pie_card, results["normal"], results["attack"])

    meter_card = card_frame(viz_row, width=520, height=340)
    meter_card.pack(side="left")
    meter_card.pack_propagate(False)
    tk.Label(meter_card, text="Threat Meter", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(pady=(15, 0))
    draw_threat_meter(meter_card, results["threat"])

    # ---- Results grid ----
    grid_card = card_frame(report)
    grid_card.pack(fill="x", pady=(0, 15))
    tk.Label(grid_card, text="Results Summary", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 10)
    )

    stats_row = tk.Frame(grid_card, bg=CARD)
    stats_row.pack(fill="x", padx=20, pady=(0, 20))

    stat_items = [
        ("Total Flows", f"{results['total']:,}", TEXT),
        ("Attack", f"{results['attack']:,}", DANGER),
        ("Normal", f"{results['normal']:,}", SUCCESS),
        ("Attack %", f"{results['attack_pct']}%", DANGER),
        ("Threat", results["threat"], threat_color(results["threat"])),
        ("Severity", results.get("severity", "Unknown"), severity_color(results.get("severity", "Unknown"))),
        ("Confidence", f"{results['confidence']}%", PRIMARY_DARK),
    ]

    for label, value, color in stat_items:
        stat = tk.Frame(stats_row, bg=BG, highlightbackground=BORDER, highlightthickness=1)
        stat.pack(side="left", expand=True, fill="both", padx=6, ipady=12)
        tk.Label(stat, text=value, bg=BG, fg=color, font=STAT_FONT).pack()
        tk.Label(stat, text=label, bg=BG, fg=SUBTEXT, font=STAT_LABEL_FONT).pack(pady=(0, 4))

    # ---- Attack Summary ----
    summary_card = card_frame(report)
    summary_card.pack(fill="x", pady=(0, 15))
    tk.Label(summary_card, text="Attack Summary", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 5)
    )
    tk.Label(
        summary_card, text=attack_summary_text(results), bg=CARD, fg=SUBTEXT, font=TEXT_FONT,
        justify="left", wraplength=1000,
    ).pack(anchor="w", padx=20, pady=(0, 18))

    # ---- Stage 2: Attack Type Breakdown ----
    breakdown = results.get("attack_breakdown") or {}
    if breakdown:
        breakdown_card = card_frame(report)
        breakdown_card.pack(fill="x", pady=(0, 15))
        tk.Label(
            breakdown_card, text="Attack Type Breakdown (Stage 2)", bg=CARD, fg=TEXT, font=BUTTON_FONT
        ).pack(anchor="w", padx=20, pady=(15, 5))
        tk.Label(
            breakdown_card,
            text=f"Most common: {results.get('top_attack_type', '-')}",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(anchor="w", padx=20, pady=(0, 5))
        draw_attack_breakdown_chart(breakdown_card, breakdown)

        # Severity per attack type, ranked the same way as the chart
        sev_row = tk.Frame(breakdown_card, bg=CARD)
        sev_row.pack(fill="x", padx=20, pady=(0, 15))
        ranked = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:6]
        for name, count in ranked:
            sev = severity_for_attack_type(name)
            chip = tk.Frame(sev_row, bg=BG, highlightbackground=BORDER, highlightthickness=1)
            chip.pack(side="left", padx=(0, 8), pady=4)
            tk.Label(
                chip, text=f"{name}  ({count:,})  -  {sev}",
                bg=BG, fg=severity_color(sev), font=SMALL_FONT, padx=10, pady=6,
            ).pack()
    elif results["attack"] > 0 and not results.get("stage2_ok"):
        breakdown_card = card_frame(report)
        breakdown_card.pack(fill="x", pady=(0, 15))
        tk.Label(
            breakdown_card,
            text="Warning: attack type classifier (Stage 2) is not loaded -- attacks were "
                 "detected but not sub-classified. Run train_attack_classifier.py to enable this.",
            bg=CARD, fg=WARNING, font=SMALL_FONT, justify="left",
        ).pack(anchor="w", padx=20, pady=15)

    # ---- Recommendations ----
    recs = results.get("recommendations") or []
    if recs:
        rec_card = card_frame(report)
        rec_card.pack(fill="x", pady=(0, 15))
        tk.Label(
            rec_card, text="Recommendations", bg=CARD, fg=TEXT, font=BUTTON_FONT
        ).pack(anchor="w", padx=20, pady=(15, 5))
        for rec in recs:
            row = tk.Frame(rec_card, bg=CARD)
            row.pack(fill="x", padx=20, pady=2, anchor="w")
            tk.Label(row, text="-", bg=CARD, fg=severity_color(results.get("severity", "Unknown")), font=TEXT_FONT).pack(
                side="left", padx=(0, 6)
            )
            tk.Label(row, text=rec, bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="left", wraplength=950).pack(
                side="left", anchor="w"
            )
        tk.Label(rec_card, text="", bg=CARD).pack(pady=(0, 6))  # bottom spacing

    # ---- Export buttons ----
    export_card = card_frame(report)
    export_card.pack(fill="x", pady=(0, 25))
    export_row = tk.Frame(export_card, bg=CARD)
    export_row.pack(pady=18)

    tk.Label(export_card, text="Export Report", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 0)
    )

    rounded_button(export_row, "Export PDF", export_pdf, width=16, height=1).pack(side="left", padx=8)
    rounded_button(export_row, "Export CSV", export_csv, bg=SUCCESS, width=16, height=1).pack(
        side="left", padx=8
    )


def threat_color(level):
    return {"LOW": SUCCESS, "MEDIUM": WARNING, "HIGH": DANGER}.get(level, TEXT)


# ============================================================
# ATTACK TYPE BREAKDOWN CHART (Stage 2)
# ============================================================


def draw_attack_breakdown_chart(parent, breakdown):
    ranked = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:10]
    labels = [name for name, _ in ranked][::-1]
    counts = [count for _, count in ranked][::-1]

    height = max(2.2, 0.4 * len(labels) + 0.8)
    fig = Figure(figsize=(9.5, height), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)

    ax.barh(labels, counts, color=DANGER)
    ax.set_xlabel("Flows", fontsize=9, color=SUBTEXT)
    ax.tick_params(axis="both", labelsize=9, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=15, pady=(0, 15), fill="x")


def draw_compact_attack_distribution_chart(parent, breakdown):
    """Smaller version of draw_attack_breakdown_chart() used only on the
    Live Capture screen, sized to sit as a compact dashboard widget next
    to the Live Flow Feed instead of a full-width panel."""
    ranked = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:6]
    labels = [name for name, _ in ranked][::-1]
    counts = [count for _, count in ranked][::-1]

    height = max(1.8, 0.4 * len(labels) + 0.6)
    fig = Figure(figsize=(3.1, height), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)

    ax.barh(labels, counts, color=DANGER, height=0.6)
    ax.set_xlabel("Flows", fontsize=7, color=SUBTEXT)
    ax.tick_params(axis="both", labelsize=7, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=(0, 10), fill="both", expand=True)


# ============================================================
# PIE CHART
# ============================================================


def draw_pie_chart(parent, normal, attack):
    fig = Figure(figsize=(4.6, 3.1), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax = fig.add_subplot(111)

    values = [normal, attack]
    labels = ["Normal", "Attack"]
    colors = [SUCCESS, DANGER]

    if sum(values) == 0:
        values = [1]
        labels = ["No Data"]
        colors = [BORDER]

    ax.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9, "color": TEXT},
    )
    ax.set_facecolor(CARD)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=5)


# ============================================================
# THREAT METER
# ============================================================


def draw_threat_meter(parent, level):
    fig = Figure(figsize=(4.6, 3.1), dpi=100)
    fig.patch.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)

    levels = ["LOW", "MEDIUM", "HIGH"]
    values = [1, 1, 1]
    colors = [SUCCESS, WARNING, DANGER]
    highlight_index = levels.index(level) if level in levels else 0

    y_pos = np.arange(len(levels))

    bar_colors = []
    for i, c in enumerate(colors):
        bar_colors.append(c if i == highlight_index else BORDER)

    ax.barh(y_pos, [10, 10, 10], color=bar_colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(levels, fontsize=10, color=TEXT)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Current Threat Level: {level}", fontsize=10, color=TEXT, pad=12)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=5)


# ============================================================
# EXPORT
# ============================================================


def export_pdf():
    results = state["results"]
    if not results:
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.join(
        REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    fig = Figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")

    ax_text = fig.add_axes([0.08, 0.55, 0.84, 0.4])
    ax_text.axis("off")

    lines = [
        "QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
        "Hybrid Quantum Neural Network -- Analysis Report",
        "",
        f"Date            : {results['date']}",
        f"Dataset         : {results['dataset']}",
        f"Total Flows     : {results['total']:,}",
        f"Normal Flows    : {results['normal']:,}",
        f"Attack Flows    : {results['attack']:,}",
        f"Attack %        : {results['attack_pct']}%",
        f"Threat Level    : {results['threat']}",
        f"Severity        : {results.get('severity', 'Unknown')}",
        f"Confidence      : {results['confidence']}%",
        f"Status          : {results['status']}",
    ]

    breakdown = results.get("attack_breakdown") or {}
    if breakdown:
        lines.append("")
        lines.append("Attack Type Breakdown (Stage 2):")
        for name, count in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {name:20s}: {count}")

    recs = results.get("recommendations") or []
    if recs:
        lines.append("")
        lines.append("Recommendations:")
        for rec in recs[:5]:
            lines.append(f"  - {rec}")

    lines += [
        "",
    ]

    ax_text.text(
        0, 1, "\n".join(lines), fontsize=10, family="monospace",
        verticalalignment="top", color="#2D2D2D",
    )

    ax_pie = fig.add_axes([0.15, 0.08, 0.7, 0.4])
    values = [results["normal"], results["attack"]]
    labels = ["Normal", "Attack"]
    colors = [SUCCESS, DANGER]
    if sum(values) == 0:
        values, labels, colors = [1], ["No Data"], [BORDER]
    ax_pie.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax_pie.set_title("Traffic Breakdown")

    fig.savefig(filename)
    messagebox.showinfo("Export Successful", f"PDF report saved to:\n{filename}")


def export_csv():
    results = state["results"]
    if not results:
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.join(
        REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    row = {
        "Date": results["date"],
        "Dataset": results["dataset"],
        "Total Flows": results["total"],
        "Normal Flows": results["normal"],
        "Attack Flows": results["attack"],
        "Attack %": results["attack_pct"],
        "Threat Level": results["threat"],
        "Severity": results.get("severity", "Unknown"),
        "Confidence": results["confidence"],
        "Status": results["status"],
        "Top Attack Type": results.get("top_attack_type") or "-",
    }

    pd.DataFrame([row]).to_csv(filename, index=False)

    breakdown = results.get("attack_breakdown") or {}
    if breakdown:
        breakdown_rows = [
            {"Attack Type": name, "Flow Count": count, "Severity": severity_for_attack_type(name)}
            for name, count in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
        ]
        with open(filename, "a", newline="") as f:
            f.write("\nAttack Type Breakdown (Stage 2)\n")
        pd.DataFrame(breakdown_rows).to_csv(filename, mode="a", index=False)

    recs = results.get("recommendations") or []
    if recs:
        with open(filename, "a", newline="") as f:
            f.write("\nRecommendations\n")
        pd.DataFrame({"Recommendation": recs}).to_csv(filename, mode="a", index=False)

    messagebox.showinfo("Export Successful", f"CSV report saved to:\n{filename}")


# ============================================================
# VIEW: LIVE CAPTURE
# ============================================================

live_widgets = {}


def view_live_capture():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=25, pady=20)

    if not SCAPY_AVAILABLE:
        card = card_frame(wrap)
        card.pack(fill="x", pady=20)
        is_macos = platform.system() == "Darwin"
        title = "Live Capture Needs macOS Setup" if is_macos else "Live Capture Needs One More Thing"
        message = (
            "Scapy could not access macOS packet capture.\n\n"
            "Make sure Scapy is installed in this virtual environment, then "
            "restart the app. Packet capture may also require administrator "
            "permission on macOS."
            if is_macos else
            "Live Capture watches your real network traffic, which needs a small,\n"
            "free driver called Npcap installed on this computer.\n\n"
            "Click the button below, install it (just click Next/Next/Finish),\n"
            "then close and reopen this app."
        )
        tk.Label(card, text=title, bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
            pady=(20, 10)
        )
        tk.Label(
            card,
            text=message,
            bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="center",
        ).pack(padx=30, pady=(0, 15))
        if not is_macos:
            rounded_button(
                card, "Install Npcap", open_npcap_download, bg=PRIMARY, width=22, height=1,
            ).pack(pady=(0, 20))
        return

    # ---------------- Controls ----------------
    controls = card_frame(wrap)
    controls.pack(fill="x", pady=(0, 15))
    row = tk.Frame(controls, bg=CARD)
    row.pack(fill="x", padx=20, pady=15)

    tk.Label(row, text="Network Connection", bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack(side="left")

    iface_choices = list_capture_interfaces()          # [(label, scapy_iface_name), ...]
    live_widgets["iface_map"] = {label: name for label, name in iface_choices}
    labels = [label for label, _ in iface_choices]

    default_label = guess_active_interface(iface_choices)

    iface_var = tk.StringVar(value=default_label if default_label else (labels[0] if labels else ""))
    iface_menu = ttk.Combobox(
        row, textvariable=iface_var, values=labels, width=45,
        state="readonly" if labels else "disabled",
    )
    iface_menu.pack(side="left", padx=10)
    live_widgets["iface_var"] = iface_var
    live_widgets["iface_menu"] = iface_menu

    rounded_button(
        row, "Refresh Interfaces", refresh_capture_interfaces, width=18, height=1,
    ).pack(side="left", padx=5)

    live_widgets["start_btn"] = rounded_button(
        row, "Start Capture", start_capture, bg=SUCCESS, width=16, height=1
    )
    live_widgets["start_btn"].pack(side="left", padx=10)
    if not labels:
        live_widgets["start_btn"].config(state="disabled")

    live_widgets["stop_btn"] = rounded_button(
        row, "Stop Capture", stop_capture, bg=DANGER, width=16, height=1
    )
    live_widgets["stop_btn"].pack(side="left", padx=5)
    live_widgets["stop_btn"].config(state="disabled")

    warn_row = tk.Frame(controls, bg=CARD)
    warn_row.pack(fill="x", padx=20)
    no_iface_label = tk.Label(
        warn_row,
        text="No active network detected. Connect to Wi-Fi or Ethernet and click Refresh.",
        bg=CARD, fg=WARNING, font=SMALL_FONT, justify="left", wraplength=760,
    )
    live_widgets["no_iface_label"] = no_iface_label
    if not labels:
        no_iface_label.pack(anchor="w", pady=(0, 10))

    live_widgets["live_status"] = tk.Label(
        controls, text="Ready - press Start Capture to begin monitoring your network",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
    )
    live_widgets["live_status"].pack(anchor="w", padx=20, pady=(0, 10))

    if not state.get("model_ok") or not state.get("scaler_ok"):
        tk.Label(
            controls,
            text="Warning: model or scaler not loaded - flows will be captured but not classified.",
            bg=CARD, fg=DANGER, font=SMALL_FONT,
        ).pack(anchor="w", padx=20, pady=(0, 10))
    elif not state.get("stage2_ok"):
        tk.Label(
            controls,
            text="Information: attack type classifier (Stage 2) not loaded - Normal/Attack detection "
                 "still runs, but detected attacks will show 'Unknown' as the type.",
            bg=CARD, fg=WARNING, font=SMALL_FONT,
        ).pack(anchor="w", padx=20, pady=(0, 10))

    # ---------------- Live stat counters ----------------
    stats_card = card_frame(wrap)
    stats_card.pack(fill="x", pady=(0, 15))
    stats_row = tk.Frame(stats_card, bg=CARD)
    stats_row.pack(pady=15)

    live_widgets["stat_total"] = make_live_stat(stats_row, "Flows Seen", "0", TEXT)
    live_widgets["stat_attack"] = make_live_stat(stats_row, "Attack Flows", "0", DANGER)
    live_widgets["stat_normal"] = make_live_stat(stats_row, "Normal Flows", "0", SUCCESS)

    # ---------------- Threat Level Indicator + Top Suspicious IPs ----------------
    insight_row = tk.Frame(wrap, bg=BG)
    insight_row.pack(fill="x", pady=(0, 15))

    threat_card = card_frame(insight_row, width=260, height=150)
    threat_card.pack(side="left", padx=(0, 15), fill="y")
    threat_card.pack_propagate(False)
    tk.Label(threat_card, text="Threat Level", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(15, 5)
    )
    threat_frame = tk.Frame(threat_card, bg=CARD)
    threat_frame.pack(fill="both", expand=True, padx=18)
    live_widgets["threat_frame"] = threat_frame

    ip_card = card_frame(insight_row, height=150)
    ip_card.pack(side="left", fill="both", expand=True)
    ip_card.pack_propagate(False)
    tk.Label(ip_card, text="Top Suspicious IPs", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=18, pady=(15, 5)
    )
    ip_frame = tk.Frame(ip_card, bg=CARD)
    ip_frame.pack(fill="both", expand=True, padx=18)
    live_widgets["ip_frame"] = ip_frame

    # ---------------- Live attack-type breakdown (Stage 2, session totals) ----------------
    type_card = card_frame(wrap)
    type_card.pack(fill="x", pady=(0, 15))
    tk.Label(type_card, text="Attack Types (this session)", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 5)
    )
    type_frame = tk.Frame(type_card, bg=CARD)
    type_frame.pack(fill="x", padx=20, pady=(0, 15))
    live_widgets["type_frame"] = type_frame
    render_live_type_breakdown()

    # ---------------- Bottom row: Live Flow Feed (left) + compact Attack
    # Distribution chart (right), side by side so both stay on screen
    # together instead of the chart pushing the feed down the page. ----
    bottom_row = tk.Frame(wrap, bg=BG)
    bottom_row.pack(fill="both", expand=True)

    # ---------------- Live table ----------------
    table_card = card_frame(bottom_row)
    table_card.pack(side="left", fill="both", expand=True, padx=(0, 15))
    tk.Label(table_card, text="Live Flow Feed", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(15, 0)
    )
    tk.Label(
        table_card, text="Double-click a row to view full flow details.",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
    ).pack(anchor="w", padx=20, pady=(0, 5))

    columns = ("time", "src", "dst", "proto", "sport", "dport", "packets", "bytes", "status", "confidence", "attack_type", "severity", "tag")
    headings = ["Time", "Source", "Destination", "Proto", "SPort", "DPort", "Pkts", "Bytes", "Status", "Conf.", "Attack Type", "Severity", "Notes"]

    style = ttk.Style()
    style.configure("Live.Treeview", rowheight=26, font=SMALL_FONT, background=CARD, fieldbackground=CARD)
    style.configure("Live.Treeview.Heading", font=SMALL_FONT)

    tree = ttk.Treeview(table_card, columns=columns, show="headings", style="Live.Treeview", height=12)
    for col, head in zip(columns, headings):
        tree.heading(col, text=head)
        tree.column(col, anchor="center", width=90)
    tree.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    tree.tag_configure("attack_row", foreground=DANGER)
    tree.tag_configure("normal_row", foreground=SUCCESS)

    tree.bind("<Double-1>", show_live_row_details)

    live_widgets["tree"] = tree
    live_widgets["row_data"] = {}

    # ---------------- Attack Distribution Chart (session totals) --------
    # Compact "dashboard widget" sized card, fixed width, docked next to
    # the Live Flow Feed instead of a full-width panel.
    dist_card = card_frame(bottom_row, width=300)
    dist_card.pack(side="left", fill="y")
    dist_card.pack_propagate(False)
    tk.Label(dist_card, text="Attack Distribution", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=15, pady=(15, 5)
    )
    dist_frame = tk.Frame(dist_card, bg=CARD)
    dist_frame.pack(fill="both", expand=True)
    live_widgets["dist_frame"] = dist_frame

    render_live_stats()



def make_live_stat(parent, label, value, color):
    col = tk.Frame(parent, bg=CARD)
    col.pack(side="left", padx=25)
    val_label = tk.Label(col, text=value, bg=CARD, fg=color, font=("Segoe UI", 18, "bold"))
    val_label.pack()
    tk.Label(col, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT).pack()
    return val_label


def render_live_stats():
    stats = state["live_stats"]
    if "stat_total" in live_widgets:
        live_widgets["stat_total"].config(text=f"{stats['total']:,}")
        live_widgets["stat_attack"].config(text=f"{stats['attack']:,}")
        live_widgets["stat_normal"].config(text=f"{stats['normal']:,}")
    render_live_type_breakdown()
    render_live_threat_indicator()
    render_live_attack_distribution()
    render_live_top_ips()


def render_live_threat_indicator():
    """Live Capture -> 'Threat Level' badge. Recomputed from the same
    session totals already tracked in state['live_stats']."""
    frame = live_widgets.get("threat_frame")
    if frame is None or not frame.winfo_exists():
        return
    for w in frame.winfo_children():
        w.destroy()

    stats = state["live_stats"]
    highest_severity = overall_severity(stats.get("attack_types") or {})
    level = compute_threat_level_full(stats.get("attack", 0), highest_severity)
    render_threat_badge(frame, level)
    tk.Label(
        frame, text=f"{stats.get('attack', 0):,} attack flow(s) this session",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="center",
    ).pack(pady=(4, 0))


def render_live_attack_distribution():
    """Live Capture -> 'Attack Distribution' bar chart. Uses the compact
    chart variant so it stays small as a dashboard widget docked next to
    the Live Flow Feed, over the live session's attack_types totals."""
    frame = live_widgets.get("dist_frame")
    if frame is None or not frame.winfo_exists():
        return
    for w in frame.winfo_children():
        w.destroy()

    attack_types = state["live_stats"].get("attack_types") or {}
    if not attack_types:
        tk.Label(
            frame, text="No attacks classified\nyet in this session.",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="center",
        ).pack(expand=True, padx=15, pady=15)
        return
    draw_compact_attack_distribution_chart(frame, attack_types)


def render_live_top_ips():
    """Live Capture -> 'Top Suspicious IPs': top 5 source IPs by attack
    count, with their highest detected severity, from
    state['live_stats']['ip_attacks']."""
    frame = live_widgets.get("ip_frame")
    if frame is None or not frame.winfo_exists():
        return
    for w in frame.winfo_children():
        w.destroy()

    ip_attacks = state["live_stats"].get("ip_attacks") or {}
    if not ip_attacks:
        tk.Label(
            frame, text="No suspicious IPs detected yet.", bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(anchor="w")
        return

    ranked = sorted(ip_attacks.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
    for ip, info in ranked:
        sev = info.get("highest_severity", "Unknown")
        row = tk.Frame(frame, bg=CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=ip, bg=CARD, fg=TEXT, font=SMALL_FONT, anchor="w").pack(side="left")
        tk.Label(
            row, text=f"{info['count']}  -  {sev}", bg=CARD, fg=severity_color(sev), font=SMALL_FONT,
        ).pack(side="right")


def render_live_type_breakdown():
    """Live Capture -> 'Attack Types (this session)' panel. Rebuilds a row
    of small chips (type, count, severity) from state['live_stats']['attack_types'].
    Safe to call even when the Live Capture view isn't currently on screen."""
    frame = live_widgets.get("type_frame")
    if frame is None or not frame.winfo_exists():
        return

    for w in frame.winfo_children():
        w.destroy()

    attack_types = state["live_stats"].get("attack_types") or {}
    if not attack_types:
        tk.Label(
            frame, text="No attacks classified yet in this session.",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(anchor="w")
        return

    ranked = sorted(attack_types.items(), key=lambda kv: kv[1], reverse=True)[:8]
    for name, count in ranked:
        sev = severity_for_attack_type(name)
        chip = tk.Frame(frame, bg=BG, highlightbackground=BORDER, highlightthickness=1)
        chip.pack(side="left", padx=(0, 8), pady=2)
        tk.Label(
            chip, text=f"{name}  ({count:,})  -  {sev}",
            bg=BG, fg=severity_color(sev), font=SMALL_FONT, padx=10, pady=6,
        ).pack()


# ---------------- Interface discovery (Windows/Npcap-native) ----------------

def open_npcap_download():
    """Beginner-friendly 'Install Npcap' button -- opens the official
    download page in the user's default browser. No commands to type."""
    try:
        webbrowser.open("https://npcap.com/#download")
    except Exception as e:
        log.warning("Could not open Npcap download page: %s", e)
        messagebox.showinfo(
            "Install Npcap",
            "Please visit https://npcap.com/#download in your browser to install Npcap.",
        )


def refresh_capture_interfaces():
    """'Refresh Interfaces' button -- re-scans network connections (e.g.
    after connecting to Wi-Fi/Ethernet) and re-selects the active one
    automatically, without needing to restart the app."""
    iface_choices = list_capture_interfaces()
    live_widgets["iface_map"] = {label: name for label, name in iface_choices}
    labels = [label for label, _ in iface_choices]

    combo = live_widgets.get("iface_menu")
    default_label = guess_active_interface(iface_choices)
    chosen = default_label if default_label else (labels[0] if labels else "")

    if "iface_var" in live_widgets:
        live_widgets["iface_var"].set(chosen)

    if combo is not None and combo.winfo_exists():
        combo.config(values=labels, state="readonly" if labels else "disabled")

    no_iface_label = live_widgets.get("no_iface_label")
    start_btn = live_widgets.get("start_btn")

    if labels:
        if no_iface_label is not None and no_iface_label.winfo_exists():
            no_iface_label.pack_forget()
        if start_btn is not None and start_btn.winfo_exists() and not state["capturing"]:
            start_btn.config(state="normal")
        if "live_status" in live_widgets and not state["capturing"]:
            live_widgets["live_status"].config(
                text="Ready - press Start Capture to begin monitoring your network", fg=SUBTEXT
            )
    else:
        if no_iface_label is not None and no_iface_label.winfo_exists():
            no_iface_label.pack(anchor="w", pady=(0, 10))
        if start_btn is not None and start_btn.winfo_exists():
            start_btn.config(state="disabled")
        if "live_status" in live_widgets and not state["capturing"]:
            live_widgets["live_status"].config(
                text="No active network detected. Connect to Wi-Fi or Ethernet and click Refresh.",
                fg=WARNING,
            )


def list_capture_interfaces():
    """Build the dropdown list from the platform's Scapy interface list.

    Returns a list of (display_label, scapy_iface_name) tuples.
    """
    results = []
    raw = []
    if get_windows_if_list is not None:
        try:
            raw = get_windows_if_list()
        except Exception as e:
            log.warning("get_windows_if_list() failed, falling back to Scapy interfaces: %s", e)

    if raw:
        for entry in raw:
            name = entry.get("name") or entry.get("description") or ""
            if not name:
                continue
            ips = entry.get("ips") or []
            ipv4s = [ip for ip in ips if ip and ":" not in ip and ip != "0.0.0.0"]
            label = f"{name}  ({ipv4s[0]})" if ipv4s else name
            results.append((label, name))

    if not results:
        # Fallback to the old raw NPF list so the dropdown is never empty,
        # even if get_windows_if_list() itself failed for some reason.
        try:
            for raw_name in get_if_list():
                results.append((raw_name, raw_name))
        except Exception as e:
            log.warning("get_if_list() fallback also failed: %s", e)

    if results:
        log.info("Detected %d capture interface(s): %s", len(results), [r[0] for r in results])
    else:
        log.warning("No capture interfaces detected -- Live Capture will need a manual interface name.")

    return results


def guess_active_interface(iface_choices):
    """Pick the adapter that's most likely the real, active connection --
    prefer Wi-Fi/Ethernet entries that have an IPv4 address, and skip
    loopback/virtual adapters (VPN, VMware, Hyper-V, Npcap Loopback)."""
    if not iface_choices:
        return ""

    skip_words = ("loopback", "virtual", "vmware", "virtualbox", "hyper-v", "bluetooth")
    prefer_words = ("wi-fi", "wifi", "wireless", "ethernet")

    scored = []
    for label, name in iface_choices:
        low = label.lower()
        if any(w in low for w in skip_words):
            continue
        score = 0
        if any(w in low for w in prefer_words):
            score += 2
        if "(" in label:  # has an IPv4 address shown -> almost certainly active
            score += 1
        scored.append((score, label))

    if not scored:
        return iface_choices[0][0]

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


# ---------------- Capture lifecycle (AsyncSniffer-based) ----------------

live_sniffer = {"instance": None}


def start_capture():
    label = live_widgets["iface_var"].get()
    iface = live_widgets.get("iface_map", {}).get(label, label)
    if not iface:
        messagebox.showerror("No Interface", "Please select a network interface first.")
        return

    state["capturing"] = True
    state["flow_store"] = {}
    state["live_stats"] = {
        "total": 0, "attack": 0, "normal": 0, "attack_types": {}, "severity_counts": {},
        "packets_captured": 0, "ip_attacks": {},
    }
    state["live_session_start"] = datetime.now().timestamp()
    state["last_packet_ts"] = None
    if "row_data" in live_widgets:
        live_widgets["row_data"] = {}
    render_live_stats()

    live_widgets["start_btn"].config(state="disabled")
    live_widgets["stop_btn"].config(state="normal")
    live_widgets["live_status"].config(
        text=f"Monitoring your network on {label}...", fg=PRIMARY_DARK
    )

    try:
        # AsyncSniffer opens its capture socket in a worker thread, where
        # permission errors otherwise do not reach this UI callback.
        probe_socket = conf.L2listen(
            iface=iface,
            filter="ip",
            promisc=True,
        )
        probe_socket.close()
        sniffer = AsyncSniffer(
            iface=iface,
            prn=packet_callback,
            store=False,
            promisc=True,
            filter="ip",
        )
        sniffer.start()
        live_sniffer["instance"] = sniffer
    except OSError as e:
        log_exception("Live capture permission/backend check", e)
        capture_failed(str(e))
        return
    except Exception as e:
        capture_failed(str(e))
        return

    root.after(EVAL_INTERVAL_MS, evaluate_live_flows)
    root.after(6000, check_capture_health)


def stop_capture():
    state["capturing"] = False

    sniffer = live_sniffer.get("instance")
    if sniffer is not None:
        try:
            sniffer.stop()
        except Exception:
            pass
        live_sniffer["instance"] = None

    live_widgets["start_btn"].config(state="normal")
    live_widgets["stop_btn"].config(state="disabled")
    live_widgets["live_status"].config(text="Capture stopped", fg=SUBTEXT)

    show_live_session_summary()


def show_live_session_summary():
    """Live Capture -> Stop Capture: modal popup summarizing the session
    that just ended. Built entirely from existing state['live_stats'] /
    state['live_session_start'] data -- no new detection logic, just a
    summary view over what the app already collected."""
    session_start = state.get("live_session_start")
    if session_start is None:
        return

    stats = state["live_stats"]
    duration_sec = max(int(datetime.now().timestamp() - session_start), 0)
    duration_str = f"{duration_sec // 60}m {duration_sec % 60}s"

    attack_types = stats.get("attack_types") or {}
    top_attack_type = max(attack_types, key=attack_types.get) if attack_types else "-"
    highest_severity = overall_severity(attack_types)

    ip_attacks = stats.get("ip_attacks") or {}
    top_ip = max(ip_attacks.items(), key=lambda kv: kv[1]["count"])[0] if ip_attacks else "-"

    summary = {
        "duration": duration_str,
        "packets": stats.get("packets_captured", 0),
        "attacks": stats.get("attack", 0),
        "normal": stats.get("normal", 0),
        "top_attack_type": top_attack_type,
        "highest_severity": highest_severity,
        "top_ip": top_ip,
        "attack_types": attack_types,
    }

    state["live_session_start"] = None

    popup = tk.Toplevel(root)
    popup.title("Live Capture Session Summary")
    popup.configure(bg=CARD)
    popup.geometry("460x680")
    popup.resizable(False, False)
    popup.transient(root)

    tk.Label(popup, text="Session Summary", bg=CARD, fg=TEXT, font=HEADING_FONT).pack(
        anchor="w", padx=20, pady=(18, 5)
    )

    def detail_row(label, value, color=TEXT):
        row = tk.Frame(popup, bg=CARD)
        row.pack(fill="x", padx=20, pady=2)
        tk.Label(row, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT, width=18, anchor="w").pack(side="left")
        tk.Label(row, text=str(value), bg=CARD, fg=color, font=SMALL_FONT, anchor="w").pack(side="left")

    detail_row("Session Duration", summary["duration"])
    detail_row("Packets Captured", f"{summary['packets']:,}")
    detail_row("Attacks Detected", f"{summary['attacks']:,}", DANGER)
    detail_row("Normal Traffic", f"{summary['normal']:,}", SUCCESS)
    detail_row("Top Attack Type", summary["top_attack_type"])
    detail_row("Highest Severity", summary["highest_severity"], severity_color(summary["highest_severity"]))
    detail_row("Top Suspicious IP", summary["top_ip"])

    tk.Frame(popup, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

    tk.Label(popup, text="Attack Distribution", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(0, 5)
    )
    chart_frame = tk.Frame(popup, bg=CARD)
    chart_frame.pack(fill="x", padx=10)
    if attack_types:
        draw_attack_breakdown_chart(chart_frame, attack_types)
    else:
        tk.Label(
            chart_frame, text="No attacks detected during this session.",
            bg=CARD, fg=SUBTEXT, font=SMALL_FONT,
        ).pack(pady=15)

    btn_row = tk.Frame(popup, bg=CARD)
    btn_row.pack(pady=15)
    rounded_button(
        btn_row, "Save Report", lambda: save_live_session_report(summary),
        bg=SUCCESS, width=16, height=1,
    ).pack(side="left", padx=8)
    rounded_button(btn_row, "Close", popup.destroy, width=14, height=1).pack(side="left", padx=8)


def save_live_session_report(summary):
    """Writes the Live Capture Session Summary to a PDF in REPORTS_DIR,
    mirroring the layout/style of the existing export_pdf()."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.join(
        REPORTS_DIR, f"live_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    fig = Figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")

    ax_text = fig.add_axes([0.08, 0.55, 0.84, 0.4])
    ax_text.axis("off")

    lines = [
        "QUANTUM NETWORK INTRUSION DETECTION SYSTEM",
        "Live Capture -- Session Summary",
        "",
        f"Date               : {datetime.now().strftime('%d-%m-%Y %H:%M')}",
        f"Session Duration   : {summary['duration']}",
        f"Packets Captured   : {summary['packets']:,}",
        f"Attacks Detected   : {summary['attacks']:,}",
        f"Normal Traffic     : {summary['normal']:,}",
        f"Top Attack Type    : {summary['top_attack_type']}",
        f"Highest Severity   : {summary['highest_severity']}",
        f"Top Suspicious IP  : {summary['top_ip']}",
    ]

    attack_types = summary.get("attack_types") or {}
    if attack_types:
        lines.append("")
        lines.append("Attack Distribution:")
        for name, count in sorted(attack_types.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {name:20s}: {count}")

    lines += [
        "",
    ]

    ax_text.text(
        0, 1, "\n".join(lines), fontsize=10, family="monospace",
        verticalalignment="top", color="#2D2D2D",
    )

    ax_chart = fig.add_axes([0.12, 0.06, 0.76, 0.42])
    if attack_types:
        ranked = sorted(attack_types.items(), key=lambda kv: kv[1], reverse=True)[:10]
        labels = [name for name, _ in ranked][::-1]
        counts = [count for _, count in ranked][::-1]
        ax_chart.barh(labels, counts, color=DANGER)
        ax_chart.set_xlabel("Flows")
        ax_chart.set_title("Attack Distribution")
    else:
        ax_chart.axis("off")
        ax_chart.text(0.5, 0.5, "No attacks detected", ha="center", va="center", color=SUBTEXT)

    fig.savefig(filename)
    messagebox.showinfo("Export Successful", f"Session report saved to:\n{filename}")


def check_capture_health():
    """One-shot watchdog fired 6s after Start. If capture is still 'on' but
    truly nothing has arrived, tell the user something specific instead of
    leaving them staring at 'Capturing...' with zero counters forever.
    Live Capture only ever shows real traffic now -- there is no automatic
    generator -- so this nudges the user to create some traffic themselves
    (e.g. browsing the web, or running an IPv4 ping/curl from Terminal)."""
    if not state["capturing"]:
        return
    if state.get("last_packet_ts") is None and "live_status" in live_widgets:
        live_widgets["live_status"].config(
            text="No IPv4 network activity detected yet. Try browsing a website, or run "
                 "'ping -c 10 8.8.8.8' in Terminal to generate IPv4 traffic to monitor.",
            fg=WARNING,
        )


def capture_failed(message):
    state["capturing"] = False
    if "live_status" in live_widgets:
        live_widgets["start_btn"].config(state="normal")
        live_widgets["stop_btn"].config(state="disabled")
        live_widgets["live_status"].config(
            text="Couldn't start monitoring -- try a different connection above", fg=DANGER
        )
    show_capture_failed_dialog()


def show_capture_failed_dialog():
    """Show platform-specific, actionable live-capture troubleshooting."""
    popup = tk.Toplevel(root)
    popup.title("Live Capture Couldn't Start")
    popup.configure(bg=CARD)
    popup.geometry("420x300")
    popup.resizable(False, False)
    popup.transient(root)

    tk.Label(popup, text="Live Capture Could Not Start", bg=CARD, fg=DANGER, font=HEADING_FONT).pack(
        anchor="w", padx=20, pady=(18, 8)
    )
    is_macos = platform.system() == "Darwin"
    message = (
        "This usually means the selected network connection isn't "
        "available right now, or macOS denied packet-capture access.\n\n"
        "Try:\n"
        "  - Click Refresh Interfaces and pick Wi-Fi or Ethernet\n"
        "  - Run the app from the active virtual environment with sudo\n"
        "  - Retry after allowing the requested macOS permission"
        if is_macos else
        "This usually means the selected network connection isn't "
        "available right now, or Npcap isn't installed yet.\n\n"
        "Try:\n"
        "  - Click Refresh Interfaces and pick your Wi-Fi/Ethernet\n"
        "  - Make sure Npcap is installed (button below)\n"
        "  - Close and reopen the app, then try again"
    )
    tk.Label(
        popup,
        text=message,
        bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="left", wraplength=370,
    ).pack(anchor="w", padx=20, pady=(0, 15))

    btn_row = tk.Frame(popup, bg=CARD)
    btn_row.pack(pady=(0, 15))
    if not is_macos:
        rounded_button(
            btn_row, "Install Npcap", open_npcap_download, bg=PRIMARY, width=18, height=1,
        ).pack(side="left", padx=6)
    rounded_button(btn_row, "Close", popup.destroy, width=12, height=1).pack(side="left", padx=6)


def packet_callback(pkt):
    if not state["capturing"] or IP not in pkt:
        return

    proto_num = pkt[IP].proto  # 6=TCP, 17=UDP
    src = pkt[IP].src
    dst = pkt[IP].dst
    length = len(pkt)

    sport = dport = 0
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport

    key = (src, dst, sport, dport, proto_num)
    now = datetime.now().timestamp()
    state["last_packet_ts"] = now
    state["live_stats"]["packets_captured"] = state["live_stats"].get("packets_captured", 0) + 1

    flow = state["flow_store"].get(key)
    if flow is None:
        flow = {
            "start": now, "last": now, "packets": 0, "bytes": 0,
            "lengths": [], "src": src, "dst": dst, "sport": sport,
            "dport": dport, "proto": proto_num,
        }
        state["flow_store"][key] = flow

    flow["packets"] += 1
    flow["bytes"] += length
    flow["lengths"].append(length)
    flow["last"] = now


def evaluate_live_flows():
    if not state["capturing"]:
        return

    now = datetime.now().timestamp()
    ready_keys = []
    stale_keys = []

    for key, flow in list(state["flow_store"].items()):
        age = now - flow["start"]
        if flow["packets"] >= MIN_PACKETS_FOR_EVAL:
            ready_keys.append(key)
        elif age > MAX_FLOW_AGE_SEC:
            stale_keys.append(key)

    for key in ready_keys:
        flow = state["flow_store"].pop(key)
        process_live_flow(flow)

    for key in stale_keys:
        state["flow_store"].pop(key, None)

    root.after(EVAL_INTERVAL_MS, evaluate_live_flows)


def process_live_flow(flow):
    duration_ms = max((flow["last"] - flow["start"]) * 1000, 1)
    lengths = flow["lengths"]

    features = {
        "protocol": flow["proto"],
        "src_port": flow["sport"],
        "dst_port": flow["dport"],
        "flow_duration": duration_ms,
        "total_packets": flow["packets"],
        "total_bytes": flow["bytes"],
        "min_packet_length": min(lengths),
        "max_packet_length": max(lengths),
        "avg_packet_length": sum(lengths) / len(lengths),
        "packets_per_second": flow["packets"] / (duration_ms + 1),
        "bytes_per_second": flow["bytes"] / (duration_ms + 1),
    }

    status = "UNKNOWN"
    confidence = 0.0
    tag = ""
    attack_type = "-"
    severity = "-"

    if state["model_ok"] and state["scaler_ok"]:
        try:
            row = np.array([[features[c] for c in FEATURE_COLUMNS]])
            scaled = state["scaler"].transform(row)
            tensor = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                output = state["model"](tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

            is_attack = int(pred.item()) == 1
            status = "ATTACK" if is_attack else "NORMAL"
            confidence = round(float(conf.item()) * 100, 1)

            if is_attack:
                tag = heuristic_tag(features)
                attack_type, _ = classify_attack_type(features)
                severity = severity_for_attack_type(attack_type if attack_type != "Unknown" else tag)

                live_types = state["live_stats"].setdefault("attack_types", {})
                live_types[attack_type] = live_types.get(attack_type, 0) + 1
                sev_counts = state["live_stats"].setdefault("severity_counts", {})
                sev_counts[severity] = sev_counts.get(severity, 0) + 1

                ip_attacks = state["live_stats"].setdefault("ip_attacks", {})
                ip_entry = ip_attacks.setdefault(flow["src"], {"count": 0, "highest_severity": "Unknown"})
                ip_entry["count"] += 1
                sev_rank = {"Unknown": -1, "Low": 0, "Medium": 1, "High": 2, "Critical": 3}
                if sev_rank.get(severity, -1) > sev_rank.get(ip_entry["highest_severity"], -1):
                    ip_entry["highest_severity"] = severity

                save_alert(make_alert(
                    source="Live Capture",
                    attack_type=attack_type,
                    severity=severity,
                    detail=f"{flow['src']}:{flow['sport']} -> {flow['dst']}:{flow['dport']}",
                    confidence=confidence,
                ))

            state["live_stats"]["total"] += 1
            if is_attack:
                state["live_stats"]["attack"] += 1
            else:
                state["live_stats"]["normal"] += 1
        except Exception as e:
            status = "ERROR"
            tag = str(e)

    insert_live_row(flow, features, status, confidence, tag, attack_type, severity)
    render_live_stats()


def heuristic_tag(features):
    """Rough, rule-based guess at attack type -- NOT a model output.
    The Hybrid QNN only classifies Normal vs Attack; this just adds a
    human-readable hint on top of that binary result."""

    if features["bytes_per_second"] > DDOS_BPS_THRESHOLD:
        return "Possible DDoS / Flood (heuristic)"

    if features["dst_port"] in BRUTEFORCE_PORTS and features["total_packets"] >= BRUTEFORCE_PACKET_THRESHOLD:
        return "Possible Brute Force (heuristic)"

    if features["packets_per_second"] > SCAN_MIN_PPS and features["avg_packet_length"] < SCAN_MAX_AVG_LEN:
        return "Possible Port Scan (heuristic)"

    return "Suspicious Traffic (heuristic)"


def insert_live_row(flow, features, status, confidence, tag, attack_type="-", severity="-"):
    tree = live_widgets.get("tree")
    if tree is None:
        return

    row_tag = "attack_row" if status == "ATTACK" else ("normal_row" if status == "NORMAL" else "")
    now_str = datetime.now().strftime("%H:%M:%S")

    item_id = tree.insert(
        "", 0,
        values=(
            now_str,
            flow["src"], flow["dst"], flow["proto"], flow["sport"], flow["dport"],
            flow["packets"], flow["bytes"], status,
            f"{confidence}%" if confidence else "-", attack_type, severity, tag,
        ),
        tags=(row_tag,) if row_tag else (),
    )

    row_data = live_widgets.setdefault("row_data", {})
    row_data[item_id] = {
        "time": now_str,
        "flow": flow,
        "features": features,
        "status": status,
        "confidence": confidence,
        "attack_type": attack_type,
        "severity": severity,
        "tag": tag,
    }

    children = tree.get_children()
    if len(children) > MAX_LIVE_ROWS:
        for extra in children[MAX_LIVE_ROWS:]:
            tree.delete(extra)
            row_data.pop(extra, None)


def show_live_row_details(event):
    """Live Capture -> double-click a row: popup with the full flow detail
    (all 11 model features plus Stage 1/Stage 2 results) that the table
    itself only summarizes."""
    tree = live_widgets.get("tree")
    if tree is None:
        return
    item_id = tree.identify_row(event.y)
    if not item_id:
        return

    data = live_widgets.get("row_data", {}).get(item_id)
    if not data:
        return

    flow = data["flow"]
    features = data["features"]

    popup = tk.Toplevel(root)
    popup.title("Flow Details")
    popup.configure(bg=CARD)
    popup.geometry("420x520")
    popup.resizable(False, False)
    popup.transient(root)

    status_color = DANGER if data["status"] == "ATTACK" else (
        SUCCESS if data["status"] == "NORMAL" else SUBTEXT
    )

    tk.Label(popup, text="Flow Details", bg=CARD, fg=TEXT, font=HEADING_FONT).pack(
        anchor="w", padx=20, pady=(18, 5)
    )
    tk.Label(
        popup, text=f"{data['status']}", bg=CARD, fg=status_color, font=("Segoe UI", 15, "bold")
    ).pack(anchor="w", padx=20, pady=(0, 10))

    def detail_row(label, value, color=TEXT):
        row = tk.Frame(popup, bg=CARD)
        row.pack(fill="x", padx=20, pady=2)
        tk.Label(row, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT, width=16, anchor="w").pack(side="left")
        tk.Label(row, text=str(value), bg=CARD, fg=color, font=SMALL_FONT, anchor="w").pack(side="left")

    detail_row("Time", data["time"])
    detail_row("Source", f"{flow['src']}:{flow['sport']}")
    detail_row("Destination", f"{flow['dst']}:{flow['dport']}")
    detail_row("Protocol", flow["proto"])
    detail_row("Confidence", f"{data['confidence']}%" if data["confidence"] else "-")
    detail_row("Attack Type", data["attack_type"])
    detail_row("Severity", data["severity"], severity_color(data["severity"]) if data["severity"] != "-" else TEXT)
    detail_row("Note", data["tag"] or "-")

    tk.Frame(popup, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

    tk.Label(popup, text="Model Input Features", bg=CARD, fg=TEXT, font=BUTTON_FONT).pack(
        anchor="w", padx=20, pady=(0, 5)
    )
    for col in FEATURE_COLUMNS:
        val = features.get(col)
        if isinstance(val, float):
            val = round(val, 2)
        detail_row(col.replace("_", " ").title(), val)

    rounded_button(popup, "Close", popup.destroy, width=14, height=1).pack(pady=15)


# ============================================================
# VIEW: TRAINING RESULTS
# ============================================================


def view_training_results():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=30, pady=25)

    tk.Label(
        wrap, text="Training Results", bg=BG, fg=TEXT, font=HEADING_FONT
    ).pack(anchor="w")
    tk.Label(
        wrap,
        text="Training evidence and saved model artifacts used by this application.",
        bg=BG,
        fg=SUBTEXT,
        font=SMALL_FONT,
    ).pack(anchor="w", pady=(2, 15))

    tabs = ttk.Notebook(wrap)
    tabs.pack(fill="both", expand=True)

    overview_tab = tk.Frame(tabs, bg=BG)
    confusion_tab = tk.Frame(tabs, bg=BG)
    report_tab = tk.Frame(tabs, bg=BG)
    artifacts_tab = tk.Frame(tabs, bg=BG)
    tabs.add(overview_tab, text="Summary & Loss")
    tabs.add(confusion_tab, text="Confusion Matrix")
    tabs.add(report_tab, text="Classification Report")
    tabs.add(artifacts_tab, text="Model Artifacts")

    columns = tk.Frame(overview_tab, bg=BG)
    columns.pack(fill="both", expand=True, padx=5, pady=5)
    columns.grid_columnconfigure(0, weight=3)
    columns.grid_columnconfigure(1, weight=2)
    columns.grid_rowconfigure(0, weight=1)

    graph_card = card_frame(columns)
    graph_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
    tk.Label(
        graph_card, text="Hybrid QNN Training Loss", bg=CARD, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="w", padx=20, pady=(18, 8))

    graph_path = os.path.join("results", "training_loss.png")
    if os.path.isfile(graph_path):
        try:
            image = tk.PhotoImage(file=graph_path)
            state["training_graph_image"] = image
            tk.Label(graph_card, image=image, bg=CARD).pack(
                fill="both", expand=True, padx=15, pady=(0, 15)
            )
        except tk.TclError as exc:
            tk.Label(
                graph_card, text=f"Training graph could not be displayed:\n{exc}",
                bg=CARD, fg=DANGER, font=TEXT_FONT, justify="left",
            ).pack(anchor="w", padx=20, pady=20)
    else:
        tk.Label(
            graph_card,
            text="Training graph not found.\nRun train_qnn.py to generate results/training_loss.png.",
            bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="left",
        ).pack(anchor="w", padx=20, pady=20)

    results_card = card_frame(columns)
    results_card.grid(row=0, column=1, sticky="nsew")
    tk.Label(
        results_card, text="Recorded Training Summary", bg=CARD, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="w", padx=20, pady=(18, 12))

    recorded_results = [
        ("QNN test accuracy", "96.83%"),
        ("Independent test accuracy", "96.62%"),
        ("Precision", "98.53%"),
        ("Recall", "95.83%"),
        ("F1 score", "97.16%"),
        ("Stage 2 validation accuracy", "84.19%"),
        ("Best Stage 2 epoch", "7"),
        ("Attack rows used", "201,070"),
        ("Attack classes", "18"),
    ]
    for label, value in recorded_results:
        row = tk.Frame(results_card, bg=CARD)
        row.pack(fill="x", padx=20, pady=4)
        tk.Label(
            row, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT, anchor="w"
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            row, text=value, bg=CARD, fg=PRIMARY_DARK, font=BUTTON_FONT, anchor="e"
        ).pack(side="right")

    tk.Label(
        results_card,
        text="The QNN metrics come from the completed test run on 10,000 test samples. "
             "The Stage 2 score is validation accuracy; rare classes have fewer examples.",
        bg=CARD, fg=SUBTEXT, font=SMALL_FONT, justify="left", wraplength=350,
    ).pack(anchor="w", padx=20, pady=(18, 20))

    confusion_card = card_frame(confusion_tab)
    confusion_card.pack(fill="both", expand=True, padx=5, pady=5)
    tk.Label(
        confusion_card, text="QNN Confusion Matrix", bg=CARD, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="w", padx=20, pady=(18, 8))
    confusion_path = os.path.join("results", "confusion_matrix.png")
    if os.path.isfile(confusion_path):
        try:
            confusion_image = tk.PhotoImage(file=confusion_path)
            state["confusion_matrix_image"] = confusion_image
            tk.Label(confusion_card, image=confusion_image, bg=CARD).pack(
                expand=True, padx=15, pady=(0, 15)
            )
        except tk.TclError as exc:
            tk.Label(
                confusion_card, text=f"Confusion matrix could not be displayed:\n{exc}",
                bg=CARD, fg=DANGER, font=TEXT_FONT, justify="left",
            ).pack(anchor="w", padx=20, pady=20)
    else:
        tk.Label(
            confusion_card,
            text="Confusion matrix not found.\nRun test_qnn.py to generate results/confusion_matrix.png.",
            bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="left",
        ).pack(anchor="w", padx=20, pady=20)

    report_card = card_frame(report_tab)
    report_card.pack(fill="both", expand=True, padx=5, pady=5)
    tk.Label(
        report_card, text="QNN Classification Report", bg=CARD, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="w", padx=20, pady=(18, 8))
    report_text = tk.Text(
        report_card, bg=CARD, fg=TEXT, font=("Menlo", 11), relief="flat",
        wrap="none", padx=20, pady=10,
    )
    report_text.pack(fill="both", expand=True, padx=10, pady=(0, 15))
    report_text.insert("1.0", """              precision    recall  f1-score   support

           0       0.94      0.98      0.96      3964
           1       0.99      0.96      0.97      6036

    accuracy                           0.97     10000
   macro avg       0.96      0.97      0.96     10000
weighted avg       0.97      0.97      0.97     10000
""")
    report_text.config(state="disabled")

    artifacts_card = card_frame(artifacts_tab)
    artifacts_card.pack(fill="both", expand=True, padx=5, pady=5)
    tk.Label(
        artifacts_card, text="Saved Model Artifacts", bg=CARD, fg=TEXT, font=BUTTON_FONT
    ).pack(anchor="w", padx=20, pady=(18, 12))
    artifacts = [
        ("Hybrid QNN", os.path.isfile(os.path.join("models", "hybrid_qnn.pth"))),
        ("Stage 1 scaler", os.path.isfile(os.path.join("processed", "scaler.pkl"))),
        ("Attack classifier", os.path.isfile(os.path.join("models", "attack_classifier.pth"))),
        ("Attack classes", os.path.isfile(os.path.join("models", "attack_classes.json"))),
        ("Stage 2 scaler", os.path.isfile(os.path.join("processed", "stage2_scaler.pkl"))),
    ]
    for label, available in artifacts:
        tk.Label(
            artifacts_card, text=f"{'Available' if available else 'Missing'}  {label}",
            bg=CARD, fg=SUCCESS if available else DANGER, font=SMALL_FONT, anchor="w",
        ).pack(fill="x", padx=20, pady=4)


# ============================================================
# VIEW: HISTORY
# ============================================================


def view_history():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=30, pady=25)

    header_row = tk.Frame(wrap, bg=BG)
    header_row.pack(fill="x", pady=(0, 15))
    tk.Label(header_row, text="Analysis History", bg=BG, fg=TEXT, font=HEADING_FONT).pack(side="left")
    rounded_button(
        header_row, "Clear History", clear_history_confirm, bg=DANGER, width=14, height=1
    ).pack(side="right")

    table_card = card_frame(wrap)
    table_card.pack(fill="both", expand=True)

    columns = ("date", "dataset", "total", "attack", "normal", "attack_pct", "threat", "confidence")
    headings = ["Date", "Dataset", "Total", "Attack", "Normal", "Attack %", "Threat", "Confidence"]

    style = ttk.Style()
    style.configure("History.Treeview", rowheight=28, font=TEXT_FONT, background=CARD, fieldbackground=CARD)
    style.configure("History.Treeview.Heading", font=BUTTON_FONT)

    tree = ttk.Treeview(table_card, columns=columns, show="headings", style="History.Treeview", height=18)
    for col, head in zip(columns, headings):
        tree.heading(col, text=head)
        tree.column(col, anchor="center", width=120)

    history = load_history()
    for entry in history:
        tree.insert(
            "", "end",
            values=(
                entry.get("date", ""),
                entry.get("dataset", ""),
                f"{entry.get('total', 0):,}",
                f"{entry.get('attack', 0):,}",
                f"{entry.get('normal', 0):,}",
                f"{entry.get('attack_pct', 0)}%",
                entry.get("threat", ""),
                f"{entry.get('confidence', 0)}%",
            ),
        )

    tree.pack(fill="both", expand=True, padx=15, pady=15)

    if not history:
        tk.Label(
            table_card, text="No scans yet. Analyzed datasets will show up here.",
            bg=CARD, fg=SUBTEXT, font=TEXT_FONT,
        ).pack(pady=20)


def clear_history_confirm():
    if messagebox.askyesno("Clear History", "This will permanently delete all saved scan history. Continue?"):
        clear_history()
        switch_view("history", view_history)


# ============================================================
# VIEW: SETTINGS
# ============================================================


def view_settings():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=30, pady=25)

    tk.Label(wrap, text="Settings", bg=BG, fg=TEXT, font=HEADING_FONT).pack(anchor="w", pady=(0, 20))

    card = card_frame(wrap)
    card.pack(fill="x")

    def setting_row(label, value_widget_fn):
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=20, pady=15)
        tk.Label(row, text=label, bg=CARD, fg=TEXT, font=TEXT_FONT, width=20, anchor="w").pack(side="left")
        value_widget_fn(row)

    setting_row(
        "Theme",
        lambda row: tk.Label(row, text="Lavender (Default)", bg=CARD, fg=PRIMARY_DARK, font=BUTTON_FONT).pack(side="left"),
    )
    setting_row(
        "Logged in as",
        lambda row: tk.Label(row, text=state["user"]["name"], bg=CARD, fg=SUBTEXT, font=TEXT_FONT).pack(side="left"),
    )
    setting_row(
        "About this system",
        lambda row: rounded_button(
            row, "View About Page", lambda: switch_view("about", view_about), width=18, height=1
        ).pack(side="left"),
    )
    setting_row(
        "System Diagnostics",
        lambda row: rounded_button(
            row, "Run Diagnostics", show_diagnostics_dialog, width=18, height=1
        ).pack(side="left"),
    )
    setting_row(
        "Analysis History",
        lambda row: rounded_button(
            row, "Reset History", clear_history_confirm, bg=DANGER, width=18, height=1
        ).pack(side="left"),
    )
    setting_row(
        "Session",
        lambda row: rounded_button(row, "Logout", logout, bg=WARNING, width=18, height=1).pack(side="left"),
    )


# ============================================================
# VIEW: ABOUT
# ============================================================


def view_about():
    wrap = tk.Frame(content_frame, bg=BG)
    wrap.pack(fill="both", expand=True, padx=30, pady=25)

    tk.Label(wrap, text="About", bg=BG, fg=TEXT, font=HEADING_FONT).pack(anchor="w", pady=(0, 20))

    card = card_frame(wrap)
    card.pack(fill="x", ipady=20)

    tk.Label(
        card, text="Quantum Network Intrusion Detection System", bg=CARD, fg=TEXT,
        font=("Segoe UI", 16, "bold"),
    ).pack(pady=(15, 2))
    tk.Label(
        card, text="Hybrid Quantum Neural Network", bg=CARD, fg=PRIMARY_DARK, font=SUBTITLE_FONT
    ).pack(pady=(0, 20))

    def about_row(label, value):
        row = tk.Frame(card, bg=CARD)
        row.pack(pady=4)
        tk.Label(row, text=label, bg=CARD, fg=SUBTEXT, font=SMALL_FONT, width=16, anchor="e").pack(side="left")
        tk.Label(row, text=value, bg=CARD, fg=TEXT, font=TEXT_FONT, anchor="w").pack(side="left")

    about_row("Model", "Hybrid Quantum Neural Network")
    about_row("Frameworks", "PyTorch, PennyLane")
    about_row("Datasets", "NF-UQ-NIDS-v2, LycoS-IDS, CIC-DDoS2019")
    about_row("Input Features", "11 network flow features")

    tk.Label(
        card,
        text="This project detects malicious network traffic using a hybrid\n"
             "architecture that combines classical neural network layers with\n"
             "a variational quantum circuit for feature transformation.",
        bg=CARD, fg=SUBTEXT, font=TEXT_FONT, justify="center",
    ).pack(pady=(20, 15))


# ============================================================
# RUN APP
# ============================================================

def show_startup_notice_if_needed():
    """After all loaders have run, tell the user (once, in plain
    language) about anything that didn't load -- instead of leaving
    them to discover it later as a silent 'Unknown'/disabled feature.
    Stage 1 problems are shown as a warning (core detection affected);
    Stage 2 problems are shown as info (attack typing only)."""
    if not state["model_ok"] or not state["scaler_ok"]:
        problems = [m for m in (state.get("model_error"), state.get("scaler_error")) if m]
        messagebox.showwarning(
            "Detection model not fully loaded",
            "The app will still open, but automatic detection won't work "
            "correctly until this is fixed:\n\n" + "\n\n".join(problems),
        )
    elif not state["stage2_ok"] and state.get("stage2_error"):
        messagebox.showinfo("Attack type classifier unavailable", state["stage2_error"])


if __name__ == "__main__":
    log.info("Starting application...")
    run_system_diagnostics()

    load_users()
    load_model()
    load_scaler()
    load_attack_classifier()

    show_startup_notice_if_needed()

    login_screen()
    root.mainloop()