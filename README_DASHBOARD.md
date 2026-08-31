# AI-Powered NIDS Streamlit SOC Dashboard

This dashboard is the presentation and triage layer for the completed AI-NIDS
model stack.

## Model stack

- **Binary detector:** known-pattern `BENIGN` versus `ATTACK`
- **Family classifier:** broad operational family for supervised attacks
- **Anomaly detector:** deviation from the learned benign profile

## Input schema

A CSV must contain all 77 features listed in:

```text
reports/tables/feature_manifest.csv
```

Optional context fields improve the alert view:

```text
Flow ID
Source IP
Source Port
Destination IP
Context Destination Port
Protocol
Timestamp
Scenario
Label
BinaryLabel
```

## Run

```powershell
python .\src\08_create_dashboard_demo.py
streamlit run .\dashboard.py
```

## Triage severity

| Severity | Meaning |
|---|---|
| Critical | Supervised attack pattern and anomaly flag |
| High | Supervised attack pattern only |
| Medium | Anomaly flag only |
| Low | Neither model path raised an alert |

The severity is a deterministic triage rule, not an independently trained or
benchmarked classifier.

## MITRE ATT&CK enrichment

The dashboard uses broad family mappings to support SOC reporting. It preserves
mapping notes because one machine-learning family may correspond to more than
one ATT&CK technique. Analysts must use the original label, packet evidence,
asset role, signatures, and incident context before making a final mapping.

## Production boundary

The dashboard does not capture packets and does not calculate the 77 trained
flow features. A live deployment needs a compatible flow extractor before
inference.
