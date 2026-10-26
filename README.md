# CV-Integrity-Assurance

**Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines**

This project is a command-line foundation for a future tool that will help assess computer-vision data, models, inference outputs, anomalies, and provenance in multi-contributor workflows. It is intended for a local Windows development environment and does not require a dashboard.

## Current foundation

Implemented now:

- A Python CLI entry point in `assurance.py` with a help screen and a foundation status summary.
- SHA-256 hashing of image files using bounded chunks, Pillow-based decoding checks, and byte-identical duplicate grouping.
- YOLO detection-label validation, including line-numbered findings, empty zero-object annotations, and image-label pairing checks.
- Local JSON model baseline registration and verification, with a guard that rejects any model not verified against its registered hash.
- Integrity-gated CPU object detection using Ultralytics YOLO11 and heuristic image-quality review checks.
- A basic source-package layout, paths/settings in `config.yaml`, and safe standard-library logging to `logs/assurance.log` when possible.
- Standard-library `unittest` checks using temporary files and directories.
- Empty `.gitkeep` placeholders for the expected data, model, report, and log directories.

The default status command reports whether the expected directories are present. `check-data` performs the first implemented dataset checks. A completed check reports data findings without stopping at invalid dataset entries; operational scan failures produce a non-zero exit code.

## Planned architecture

The `src/` packages reserve space for future functionality:

- `integrity/`: dataset and artifact integrity checks.
- `model/`: model integrity verification.
- `vision/`: computer-vision inference.
- `anomaly/`: anomaly detection.
- `provenance/`: contributor and artifact provenance.
- `database/`: PostgreSQL connection and inference-evidence persistence.
- `reports/`: local PDF assurance report generation.

The CLI reads configured dataset paths, the `paths.model_registry` and `paths.reports` values, and inference/anomaly settings from `config.yaml`.

## Windows setup (PowerShell)

Install Python 3.12 (the project was specified for Python 3.12.8) and ensure the Windows Python launcher `py` is available. From the project root:

```powershell
py -3.12 --version
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable in the virtual environment directly, or adjust the execution policy for your user according to your organization's guidance.

Pillow decodes input images and calculates simple image-quality metrics. Ultralytics supplies the local YOLO11 runtime. Psycopg 3 with its binary distribution connects to PostgreSQL. Install project dependencies in the virtual environment:

```powershell
py -3.12 -m pip install -r requirements.txt
```

## CLI usage

Run from the project root:

```powershell
py -3.12 assurance.py --help
py -3.12 assurance.py
py -3.12 assurance.py check-data
py -3.12 assurance.py register-model --id demo-model --path models/demo-model.bin
py -3.12 assurance.py verify-model --id demo-model
py -3.12 assurance.py infer --id yolo-model --image dataset/images/sample.jpg
py -3.12 assurance.py init-db
py -3.12 assurance.py assess --id yolo11n --image dataset/images/bus.jpg
py -3.12 assurance.py demo-model-tampering --id yolo11n
py -3.12 assurance.py demo-data-integrity
```

The default command prints the project name, whether its expected directories are present, and the capabilities planned for later milestones. `check-data` uses `paths.images` and `paths.labels` from `config.yaml`; optional `--images-dir` and `--labels-dir` arguments override either directory. It does not require PostgreSQL, credentials, model weights, or internet access.

Model registry entries are stored at `paths.model_registry` (currently `models/model_registry.json`). Relative model paths stored in the registry are resolved from the project root, regardless of the caller's current working directory; absolute model paths remain supported. Register a local model file once to record its SHA-256 baseline. Re-registering an existing ID fails unless `--replace` is explicitly supplied. Both model commands accept `--registry PATH` to override the configured registry. Registration only records a local baseline: hash agreement indicates unchanged bytes since that baseline, not authenticity of the source, accuracy, or safety. `SUSPICIOUS` means the model bytes changed and is a critical integrity failure; `REVIEW REQUIRED` means the model cannot be safely verified. The reusable `ensure_model_verified` guard blocks inference use for either state.

### Local YOLO inference

Download the official pretrained YOLO11 nano detection weights (`yolo11n.pt`) from the [Ultralytics assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) into the configured `models/` folder. The weights are not included in the repository; `.pt` files are ignored by Git. The first download requires internet access. After it is downloaded, inference runs locally on CPU and does not use a cloud inference service.

Register the local file as a baseline, then infer:

```powershell
py -3.12 assurance.py register-model --id yolo-model --path models/yolo11n.pt
py -3.12 assurance.py infer --id yolo-model --image dataset/images/sample.jpg
```

Registration is a separate explicit trust decision. Inference verifies the registered SHA-256, copies the file to a temporary private snapshot, checks the snapshot against the same baseline, then loads that snapshot only if verification remains `VERIFIED`. It never auto-registers changed weights. `--conf` optionally overrides the configured confidence threshold, and inference is CPU-only by default. Detection confidence is a model score, not proof of a correct prediction.

`config.yaml` contains initial image-quality heuristic thresholds for dimensions, aspect ratio, mean brightness, and grayscale Laplacian variance (blur estimate). Findings mark a run `REVIEW REQUIRED` while allowing the user to inspect the result; they are not proof of malicious activity. These thresholds need calibration against the target dataset. Invalid/unreadable images block inference.

### PostgreSQL provenance evidence

The database defaults are `localhost:5433`. Copy `.env.example` to `.env`, replace the database and user placeholders, and set a password only if the local PostgreSQL authentication method requires one. `.env` is excluded from Git. Environment variables override values in `.env`; supported keys are `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`. The application never prints or stores the password.

Initialize the schema explicitly after configuring PostgreSQL:

```powershell
py -3.12 assurance.py init-db
```

This creates `inference_provenance` only when it does not already exist; it does not drop tables or erase rows. Inference attempts create a unique run ID and, when PostgreSQL settings are available, insert image/model hashes and verification status, configuration, detections, image metrics, outcome, and reasons. The CLI prints the run ID and whether evidence was saved. If no database is configured, persistence is reported as skipped; if the connection or insert fails, persistence is reported as not saved while the actual inference result is retained. A blocked model attempt is also recorded when the database is reachable, without loading or executing that model.

Provenance is a record of data observed by this application, not proof that a model source is authentic or that a prediction is objectively correct. Database availability does not bypass model verification.

### End-to-end assessment and PDF

Run a complete integrity-gated assessment using the existing registered model:

```powershell
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python assurance.py assess --id yolo11n --image dataset/images/bus.jpg
```

Each completed assessment writes a uniquely named PDF under `paths.reports` (currently `reports/`) and prints its exact path. A report is still generated if PostgreSQL is unavailable, provided local assessment evidence can be generated. The output separately shows the PostgreSQL evidence status and provenance run ID; skipped or failed persistence is never described as saved.

Assessment statuses follow these rules: `SUSPICIOUS` takes precedence when the registered model hash mismatches; otherwise `REVIEW REQUIRED` applies if image/model evidence is incomplete, an image-quality heuristic triggers, inference fails, or PostgreSQL evidence could not be persisted; `VERIFIED` requires successful model verification and inference, no anomaly warnings, and successful PostgreSQL persistence. Database unavailability does not indicate tampering. These are rule-based findings, not proof of a model's original authenticity, safety, accuracy, or prediction correctness. A matching hash only shows that model bytes match the locally registered baseline. Anomaly thresholds are initial heuristics and require dataset-specific calibration.

## Reproducible evaluator demonstrations

From the activated `.venv312`, run:

```powershell
python assurance.py verify-model --id yolo11n
python assurance.py assess --id yolo11n --image dataset/images/bus.jpg
Get-ChildItem reports\assurance-*.pdf | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python assurance.py demo-model-tampering --id yolo11n
python assurance.py demo-data-integrity
```

`demo-model-tampering` requires an intact registered source model. It copies the file into a temporary directory, creates a temporary baseline, modifies only that copy, and shows the expected/actual hashes and `SUSPICIOUS` status. It then uses a mocked loader and confirms inference is blocked before loading. The temporary directory is removed and the source hash is checked for change. `demo-data-integrity` creates valid/corrupt/duplicate images, a malformed label, and an image missing its label in a temporary directory; it prints the findings and cleans up all demo files. Neither demo writes to the real dataset or model files.

`VERIFIED` means the model hash matched, inference succeeded, no review-triggering image heuristic was found, and PostgreSQL evidence persisted. `REVIEW REQUIRED` indicates incomplete evidence, an anomaly, an inference failure, or unavailable database persistence. `SUSPICIOUS` indicates the model bytes differ from its registered baseline, and loading is blocked. These are rule-based results: hash matching confirms unchanged bytes since registration, not original authenticity, model accuracy, safety, or prediction correctness.

Supported image extensions are BMP, GIF, JPEG, PNG, TIFF, and WebP. Pillow decodes each candidate; unsupported extensions, empty files, and files that cannot be decoded are reported invalid. Duplicate groups are based only on equal SHA-256 hashes (byte-identical files), not visual similarity. YOLO labels must contain five fields per non-empty line: a non-negative integer class ID and finite normalized box values. Empty label files are valid zero-object annotations.

## Tests

Run the standard-library tests from the project root:

```powershell
py -3.12 -m unittest discover -s tests -v
```

## Configuration, secrets, and logging

- `config.yaml` contains project-relative paths and image/inference thresholds.
- `.env.example` gives placeholder PostgreSQL settings for localhost port `5433`. Copy it to `.env` and replace required placeholders; never commit `.env`.
- The CLI attempts to write logs to `logs/assurance.log`. If file logging cannot be configured, it falls back to standard error and continues starting.

## Current limitations

This milestone adds image/label checks, local model hash-baseline verification, CPU YOLO11 detection, image-quality heuristics, optional PostgreSQL provenance persistence, and local PDF reports. No training or fine-tuning is implemented. No dataset or model weights are included in source control; automated tests use synthetic files and mocked inference/database connections.
