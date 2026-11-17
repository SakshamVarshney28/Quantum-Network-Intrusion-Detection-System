
# STAGE 2 - ATTACK TYPE DEFINITIONS
import re
# Canonical display order/names -- exactly as specified for the project.
ATTACK_CLASSES = [
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "Slowloris",
    "SlowHTTPTest",
    "Port Scan",
    "Botnet",
    "SSH Brute Force",
    "FTP Brute Force",
    "SQL Injection",
    "XSS",
    "Infiltration",
    "Heartbleed",
    "DNS Flood",
    "SYN Flood",
    "ICMP Flood",
]


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

# Labels that mean "this row is not an attack" and should be dropped
# before training Stage 2 (Stage 1 already handles Normal vs Attack).
_BENIGN_ALIASES = {"benign", "normal", "normaltraffic", "background"}

_LABEL_ALIASES = {
    # DDoS
    "ddos": "DDoS",
    "ddosattack": "DDoS",
    "ddosattackhoic": "DDoS",
    "ddosattackloic": "DDoS",
    "ddosattackloicudp": "DDoS",
    "ddosattackloichttp": "DDoS",
    "udpflood": "DDoS",
    "udplag": "DDoS",

    # DoS Hulk
    "doshulk": "DoS Hulk",
    "hulk": "DoS Hulk",

    # DoS GoldenEye
    "dosgoldeneye": "DoS GoldenEye",
    "goldeneye": "DoS GoldenEye",

    # Slowloris
    "dosslowloris": "Slowloris",
    "slowloris": "Slowloris",

    # SlowHTTPTest
    "dosslowhttptest": "SlowHTTPTest",
    "slowhttptest": "SlowHTTPTest",
    "slowhttp": "SlowHTTPTest",

    # Port Scan
    "portscan": "Port Scan",
    "scanning": "Port Scan",
    "nmap": "Port Scan",

    # Botnet
    "bot": "Botnet",
    "botnet": "Botnet",

    # SSH Brute Force
    "sshpatator": "SSH Brute Force",
    "sshbruteforce": "SSH Brute Force",
    "sshbrute": "SSH Brute Force",

    # FTP Brute Force
    "ftppatator": "FTP Brute Force",
    "ftpbruteforce": "FTP Brute Force",
    "ftpbrute": "FTP Brute Force",
    "bruteforce": "FTP Brute Force",  # generic "brute force" label fallback

    # SQL Injection
    "sqlinjection": "SQL Injection",
    "webattacksqlinjection": "SQL Injection",

    # XSS
    "xss": "XSS",
    "webattackxss": "XSS",
    "crosssitescripting": "XSS",

    # Infiltration
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",  # common misspelling in CICIDS2018

    # Heartbleed
    "heartbleed": "Heartbleed",

    # DNS Flood
    "dnsflood": "DNS Flood",
    "dns": "DNS Flood",
    "drdosdns": "DNS Flood",  # CIC-DDoS2019 reflection label

    # SYN Flood
    "synflood": "SYN Flood",
    "tcpsynflood": "SYN Flood",
    "syn": "SYN Flood",  # CIC-DDoS2019 label

    # ICMP Flood
    "icmpflood": "ICMP Flood",
    "icmp": "ICMP Flood",

    # Port Scan (NF-UQ-NIDS-v2's ToN-IoT slice uses "scanning")
    "recon": "Port Scan",
    "reconnaissance": "Port Scan",

    # DDoS -- CIC-DDoS2019's other reflection/amplification & flood
    # subtypes that don't map to a more specific canonical category here.
    "drdosldap": "DDoS",
    "drdosmssql": "DDoS",
    "drdosnetbios": "DDoS",
    "drdosntp": "DDoS",
    "drdossnmp": "DDoS",
    "drdosssdp": "DDoS",
    "drdosudp": "DDoS",
    "drdostftp": "DDoS",
    "tftp": "DDoS",
    "portmap": "DDoS",
    "webddos": "DDoS",
    "udp": "DDoS",
    "mssql": "DDoS",
    "ldap": "DDoS",
    "netbios": "DDoS",
    "ntp": "DDoS",
    "snmp": "DDoS",
    "ssdp": "DDoS",

    "webattackbruteforce": "FTP Brute Force",

    # NF-UQ-NIDS-v2 merges several source datasets (UNSW-NB15, BoT-IoT,
    # ToN-IoT, CSE-CIC-IDS2018) whose "Attack" category column uses
    # broader family names than the 16-class list. Map the ones with an
    # obvious match; the rest (exploits, fuzzers, generic, shellcode,
    # worms, backdoor, mitm, password, ransomware, theft, injection) are
    # left unmapped on purpose -- they become their own classes instead
    # of being force-fit into an unrelated canonical bucket.
    "dos": "DDoS",
    "password": "SSH Brute Force",
}


def _normalize(raw):
    return re.sub(r"[^a-z0-9]", "", str(raw).strip().lower())


def normalize_label(raw_label):
   
    key = _normalize(raw_label)
    if key in _BENIGN_ALIASES:
        return None
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    return str(raw_label).strip().title()