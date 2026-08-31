# AI-Powered Network Intrusion Detection System

A hybrid network intrusion detection prototype that combines supervised machine learning, attack-family classification, benign-trained anomaly detection, Suricata rule-based analysis, MITRE ATT&CK enrichment, and source-level cross-flow behavioral correlation.

## Project Overview

The project analyzes CICIDS/CICFlowMeter-style network-flow records and produces security-oriented predictions and evidence for SOC triage. It was developed in stages:

1. Audit and align the source datasets.
2. Prepare a reproducible 77-feature modeling population.
3. Compare binary intrusion-detection models.
4. Evaluate known-attack performance with group-aware data splits.
5. stress-test generalization by holding out complete traffic scenarios.
6. Classify detected attacks into operational attack families.
7. Add a benign-trained Isolation Forest for anomaly detection.
8. Build a Streamlit SOC dashboard.
9. Integrate Suricata in an isolated Kali/Ubuntu VMware laboratory.
10. Capture benign and controlled port-scan traffic.
11. Compare Suricata and machine-learning predictions on the same traffic.
12. Add cross-flow correlation to detect aggregate scanning behavior.

## Key Capabilities

- Binary classification: `BENIGN` or `ATTACK`
- Attack-family enrichment:
  - Denial of Service
  - Port Scan
  - Brute Force
  - Web Attack
  - Other Malicious
- Benign-trained anomaly detection
- MITRE ATT&CK enrichment
- Streamlit SOC dashboard
- Suricata default-rule and tuned-rule comparison
- Safe attack simulation in an isolated VMware lab
- Source-level rolling correlation for port-scan detection
- Evidence ledgers, summaries, figures, and SHA-256 manifests

## System Architecture

```text
CICIDS/CICFlowMeter-style flow record
        |
        +--> 77-feature compatibility layer
        |
        +--> Group-aware supervised binary detector
        |       +--> BENIGN
        |       +--> ATTACK
        |              |
        |              +--> Attack-family classifier
        |
        +--> Benign-trained Isolation Forest
        |       +--> Within learned benign profile
        |       +--> Anomalous
        |
        +--> Source-level rolling correlation
                +--> Source IP
                +--> SYN-bearing flow count
                +--> Unique destination-port count
                +--> Rolling 10-second window
                +--> PORT_SCAN / T1595
```

Suricata operates as a separate rule-based detection layer against the same captured traffic.

## Dataset and Preprocessing

The preprocessing pipeline processed:

- 8 machine-learning CSV files
- 8 generated-flow context CSV files
- 2,830,743 valid aligned rows
- 77 predictive features after removal of the duplicate `Fwd Header Length.1` field

The pipeline:

- Preserves the original raw files
- Replaces infinity values with missing values
- Retains missing numeric values for training-only imputation
- Normalizes the multiclass label
- Creates a binary target
- Creates an operational attack-family target
- Verifies row counts and shared-column alignment across source collections

## Detection Components

### 1. Supervised Binary Detector

The binary stage determines whether each flow resembles a learned benign or attack pattern. Candidate models included:

- Dummy prior baseline
- Logistic Regression using SGD
- Random Forest
- HistGradientBoosting

The selected model family is HistGradientBoosting.

### 2. Attack-Family Classifier

The attack-family stage runs only after the binary stage predicts `ATTACK`. The selected family model is a Random Forest classifier covering broad operational families rather than exact malware attribution.

### 3. Isolation Forest

The anomaly detector is trained on benign traffic. It flags deviations from the learned benign profile but does not treat an anomaly as proof of malicious activity.

### 4. Cross-Flow Behavioral Correlation

The cross-flow detector supplements the per-flow models. It correlates traffic from the same source over a rolling time window.

The tested laboratory condition is:

```text
At least 20 SYN-bearing flows
to at least 20 unique destination ports
from one source
within 10 seconds
```

The detector emits one de-duplicated source-level alert and maps the activity to:

- MITRE ATT&CK tactic: Reconnaissance
- Technique: T1595
- Technique name: Active Scanning
- Severity: High

## Evaluation Summary

The dashboard preserves separate metrics for different evaluation designs because random flow-level performance is not equivalent to novel-scenario generalization.

| Evaluation | Key result |
|---|---:|
| Grouped known-attack binary recall | 99.92% |
| Attack-family macro recall | 98.68% |
| Complete-scenario supervised recall | 15.45% |
| Complete-scenario anomaly recall | 35.74% |

The complete-scenario tests demonstrate that performance falls substantially when entire traffic scenarios are excluded from training.

## Live Laboratory

### Environment

- Kali Linux VM: controlled traffic generation
- Ubuntu VM: Suricata sensor and target
- VMware host-only network: `192.168.244.0/24`
- Kali source: `192.168.244.10`
- Ubuntu target: `192.168.244.20`
- Suricata interface: `ens37`

Testing was limited to systems owned and controlled within the isolated lab.

### Benign Baseline

The benign scenario included ping and Python HTTP-server traffic.

Flow-level result:

- 5 supported HTTP flows
- 5 `BENIGN` predictions
- 0 supervised attack predictions
- 0 anomaly flags
- No cross-flow scan pattern

Suricata produced 34 informational or decoder-related records:

- 4 ICMP ping notices
- 5 Python SimpleHTTP server-banner notices
- 25 TCP checksum decoder records

### Controlled Port Scan

The controlled scan generated:

- 502 SYN probes
- 502 unique destination ports
- 502 CICFlowMeter-compatible flow rows

Per-flow machine-learning result:

- 502 `BENIGN` predictions
- 0 `ATTACK` predictions
- 0 anomaly flags
- Attack-family classifier not invoked

Default Suricata result:

- 502 flows processed
- 0 emitted scan alerts
- 3 suppressed alerts recorded in engine statistics

Lab-tuned Suricata result:

- 24 threshold notification events
- One controlled scan scenario
- IDS action: allowed and logged, not blocked

Cross-flow correlation result:

- `PORT_SCAN`
- One de-duplicated source-level alert
- High severity
- T1595 Active Scanning
- Maximum unique destination ports observed in a 10-second window: 276

The benign baseline remained `NO_SCAN_PATTERN`, with one unique destination port and a maximum of four SYN-bearing flows in a window.

## Detection-Layer Comparison

| Detection layer | Benign baseline | Controlled port scan |
|---|---|---|
| Default Suricata | Informational/decoder records only | No emitted scan alert |
| Lab-tuned Suricata | Not applicable | Detected |
| Supervised per-flow ML | Correctly benign | Missed |
| Attack-family classifier | Not invoked | Not invoked |
| Isolation Forest | No anomaly | Missed |
| Cross-flow behavioral correlation | No scan pattern | Detected |

The experiment shows why multiple detection layers are valuable. The per-flow models did not see the aggregate behavior of one source contacting hundreds of ports. Suricata rule tuning and source-level correlation exposed that pattern.

## Dashboard

The Streamlit application includes:

- CSV upload and local demonstration data
- Input preview
- Binary and anomaly outputs
- Attack-family distributions
- Severity summaries
- Source-IP context
- MITRE ATT&CK enrichment
- SOC alert queue
- Per-flow inspection
- Validated model-evidence page
- Architecture and limitations page
- Cross-flow correlation page
- Consolidated Suricata-versus-ML comparison

Run the application with the main dashboard entrypoint:

```powershell
python -m streamlit run .\dashboard.py
```

When the dashboard entrypoint is located elsewhere, replace `.\dashboard.py` with the discovered path.

## Repository Structure

```text
AI-NIDS-Project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── live_lab/
│       ├── raw/
│       └── model_ready/
├── models/
│   ├── binary/
│   ├── attack_family/
│   └── anomaly/
├── notebooks/
├── pages/
│   └── 4_Cross_Flow_Correlation.py
├── reports/
│   ├── evidence/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── src/
├── dashboard.py
├── requirements.txt
└── README.md
```

The actual dashboard or pages directory may be under `src/` depending on the local entrypoint layout.

## Reproducing the Main Results

### Activate the Windows environment

```powershell
Set-Location C:\Users\esumi\AI-NIDS-Project
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Prepare live-lab flow files

```powershell
python .\src\10_prepare_live_lab_flows.py
```

### Run cross-flow correlation

```powershell
python .\src\11_cross_flow_scan_correlation.py `
    --input .\data\live_lab\model_ready\benign_baseline_dashboard.csv `
    --input .\data\live_lab\model_ready\portscan_rate_limited_dashboard.csv `
    --window-seconds 10 `
    --minimum-unique-ports 20 `
    --minimum-syn-flows 20
```

### Start the dashboard

```powershell
python -m streamlit run .\dashboard.py
```

## Evidence Locations

Important evidence is stored under:

```text
reports/
├── evidence/
│   ├── dashboard/
│   └── suricata/
├── figures/
│   └── final_dashboard/
├── predictions/
│   └── live_lab/
└── tables/
```

Principal result files include:

```text
reports/tables/live_lab_feature_compatibility.csv
reports/tables/live_lab_cross_flow_summary.csv
reports/tables/final_live_lab_detection_comparison.csv
reports/predictions/live_lab/benign_prediction_ledger.csv
reports/predictions/live_lab/portscan_prediction_ledger.csv
reports/predictions/live_lab/cross_flow_scan_alerts.csv
reports/evidence/suricata/benign_alerts.json
reports/evidence/suricata/portscan_default_alerts.json
reports/evidence/suricata/portscan_tuned_alerts.json
```

## Limitations

- The training data comes from one historical dataset and may not represent current enterprise traffic.
- Very high internal known-attack scores do not imply equally high performance on new environments.
- The community CICFlowMeter extractor may not reproduce every feature exactly as the original dataset-generation implementation.
- The binary gate prevents attack-family classification when Stage 1 predicts `BENIGN`.
- The Isolation Forest did not detect the controlled external scan.
- The tested cross-flow thresholds are laboratory settings and require broader benign-traffic validation.
- The tuned Suricata rule is lab-specific.
- The current workflow captures traffic, extracts flows, and performs inference; it is not yet a fully automated continuous packet-to-alert service.
- MITRE ATT&CK mappings are triage enrichment and not definitive attribution.
- Production use requires drift monitoring, secure logging, access control, alert suppression, retraining, and analyst feedback.

## Security and Ethical Boundaries

All attack simulation must be restricted to isolated systems that are owned or explicitly authorized for testing. The controlled scan in this project targeted only the Ubuntu VM inside the private VMware laboratory.

Do not run scanning, flooding, or exploit tools against external systems without written authorization.

## Future Work

- Automate capture-to-flow-to-inference processing
- Validate on current external enterprise traffic
- Re-extract training traffic with the same live inference toolchain
- Add rolling source, destination, and asset-context features
- Tune cross-flow rules against a larger benign baseline
- Add alert suppression and deduplication policies
- Add model and feature-drift monitoring
- Deploy inference through a secured API
- Add analyst feedback and retraining workflows
- Evaluate Zeek-based enrichment
- Test additional controlled scenarios such as brute force and web attacks

## Project Status

The portfolio-grade prototype is complete for:

- Model development and comparison
- Group-aware and scenario-holdout evaluation
- Attack-family classification
- Anomaly detection
- Streamlit dashboarding
- Suricata integration
- Safe live-lab capture
- Same-traffic Suricata-versus-ML comparison
- MITRE ATT&CK enrichment
- Source-level cross-flow correlation

Continuous automated real-time inference remains a production-hardening phase rather than a completed claim.
