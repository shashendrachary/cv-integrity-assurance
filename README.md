# CV-Integrity-Assurance

**Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines**

This project is a command-line tool for assessing computer-vision data, model integrity, inference outputs, image-quality anomalies, and provenance in multi-contributor workflows. It is intended for local Windows development and does not require a dashboard.

## Implemented capabilities

Implemented now:

- A Python CLI entry point in `assurance.py` with a help screen and a foundation status summary.
- SHA-256 hashing of image files using bounded chunks, Pillow-based decoding checks, and byte-identical duplicate grouping.
- YOLO detection-label validation, including line-numbered findings, empty zero-object annotations, and image-label pairing checks.
- Local JSON model baseline registration and verification, with a guard that rejects any model not verified against its registered hash.
- Integrity-gated CPU object detection using Ultralytics YOLO11 and heuristic image-quality review checks.
- A basic source-package layout, paths/settings in `config.yaml`, and safe standard-library logging to `logs/assurance.log` when possible.
- Standard-library `unittest` checks using temporary files and directories.

- Empty `.gitkeep` placeholders for the expected data, model, report, and log directories.

## CLI overview

The default status command reports project and directory information. The CLI supports dataset validation, model baseline registration and verification, integrity-gated image inference, single-image assessment, batch image assessment, sampled video-frame assessment, PDF/JSON reporting, model-tampering demonstrations, and optional PostgreSQL provenance persistence. A completed dataset scan reports findings without stopping at individual invalid entries; operational scan failures produce a non-zero exit code.


## Project architecture

The `src/` packages organize the implemented functionality:


- `integrity/`: dataset and artifact integrity checks.
- `model/`: model integrity verification and protected model loading.
- `vision/`: computer-vision inference.
- `anomaly/`: image-quality review heuristics.
- `provenance/`: project attribution and inference evidence.
- `database/`: PostgreSQL connection, schema migration, and evidence persistence.
- `reports/`: local PDF assurance reports, including consolidated batch and video reports.


The CLI reads project identity, configured dataset paths, the `paths.model_registry` and `paths.reports` values, and inference/anomaly settings from `config.yaml`.

### Project configuration context

The root `config.yaml` remains the default project configuration and uses the stable identifiers `organization_id: local` and `project_id: default`. Resource paths are resolved relative to the directory containing the selected configuration; absolute paths remain supported. To select another project, provide a configuration file with the same `project` and `paths` sections and pass `--project-config PATH` to a project-resource command:

```yaml
project:
  organization_id: org-blue
  project_id: road-inspection
paths:
  images: dataset/images
  labels: dataset/labels
  model_registry: models/model_registry.json
  reports: reports
```

For example, `python assurance.py check-data --project-config projects/road/config.yaml` selects that configuration and resolves its relative resource paths from `projects/road/`. The option is supported by `check-data`, `register-model`, `verify-model`, `infer`, `assess`, `assess-batch`, `assess-video`, and `demo-model-tampering`; existing commands without it continue using the root configuration. Relative `--registry`, `--path`, `--image`, `--video`, `--images-dir`, and `--labels-dir` overrides use the selected configuration directory; absolute overrides remain absolute. The `init-db` and temporary `demo-data-integrity` commands do not use project resources and do not accept this option.

The project context validates identifiers and groups resource paths for CLI selection and attribution. It does not authenticate users or restrict filesystem access. Project IDs in provenance are attribution, not a database security boundary.

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
py -3.12 assurance.py assess-batch --id yolo11n --images-dir dataset/images
py -3.12 assurance.py demo-model-tampering --id yolo11n
py -3.12 assurance.py demo-data-integrity
py -3.12 assurance.py assess-video --id yolo11n --video dataset/videos/vtest.avi --interval-seconds 2
```

The default command prints the project name, whether its expected directories are present, and the available CLI capabilities. `check-data` uses `paths.images` and `paths.labels` from `config.yaml`; optional `--images-dir` and `--labels-dir` arguments override either directory. It does not require PostgreSQL, credentials, model weights, or internet access.

Model registry entries are stored at `paths.model_registry` (currently `models/model_registry.json`). Relative model paths stored in the registry are resolved from the project root, regardless of the caller's current working directory; absolute model paths remain supported. Register a local model file once to record its SHA-256 baseline. Re-registering an existing ID fails unless `--replace` is explicitly supplied. Both model commands accept `--registry PATH` to override the configured registry. Registration only records a local baseline: hash agreement indicates unchanged bytes since that baseline, not authenticity of the source, accuracy, or safety. `SUSPICIOUS` means the model bytes changed and is a critical integrity failure; `REVIEW REQUIRED` means the model cannot be safely verified. The reusable `ensure_model_verified` guard blocks inference use for either state.

### Local YOLO inference

Download the official pretrained YOLO11 nano detection weights (`yolo11n.pt`) from the [Ultralytics assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) into the configured \models/`` folder. The weights are not included in the repository; `.pt` files are ignored by Git. The first download requires internet access. After it is downloaded, inference runs locally on CPU and does not use a cloud inference service.

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

`init-db` retains its legacy behavior: it creates the original `inference_provenance` table only when absent and never upgrades an existing table. To enable project-scoped persistence, back up the intended database and explicitly run:

```powershell
py -3.12 assurance.py migrate-db
```

The migration creates or upgrades the provenance schema and assigns existing unscoped rows to `local/default`; it is never run by `init-db` or ordinary CLI commands. New inference and assessment evidence includes the selected project's organization and project IDs. When the scoped schema is present, persistence transactionally adds a previously unseen organization/project identity without replacing existing registrations, then inserts the evidence. Persistence against a legacy schema that has not been migrated is reported as not saved; the local inference/report behavior remains available. Inference attempts create a unique run ID and, when PostgreSQL settings are available and the scoped schema is present, insert image/model hashes and verification status, configuration, detections, image metrics, outcome, and reasons. A blocked model attempt is also recorded when the database is reachable, without loading or executing that model.

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

### Batch image assessment

Assess every supported image beneath a directory, recursively and in deterministic path order:

```powershell
python assurance.py assess-batch --id yolo11n --images-dir dataset/images
```

The batch command reuses the single-image assessment workflow and model-integrity checks, but does not create per-image PDFs. After all images have been attempted, it writes one consolidated PDF and a matching JSON sidecar containing detection details. The PDF summarizes discovered and processed images, successful/review-required/failed outcomes, model verification, image hashes and properties, quality flags, aggregate detections, and findings. It prints each image's status and the final report paths, then returns a non-zero exit code if any image requires review or fails. `--images-dir` is required; relative paths use the selected project configuration directory. The command also accepts `--conf`, `--registry`, and `--project-config`, with the same override behavior as `assess`.

### Video frame assessment

Assess frames sampled once per second by default:

```powershell
python assurance.py assess-video --id yolo11n --video dataset/videos/inspection.mp4 --interval-seconds 2.0
```

Video decoding uses OpenCV. The command processes frames sequentially without loading the entire video into memory, and each sampled frame runs through the same model-integrity checks and assessment workflow as a still image. After sampling, it generates exactly one consolidated PDF and a matching JSON sidecar with frame-level detection details. The report/provenance identifies the original video path and records the sampled frame index, time, interval, and frame-hash scope; temporary frame files are removed after assessment. The command does not connect to PostgreSQL. It accepts `--conf`, `--registry`, and `--project-config`; the interval must be finite and greater than zero. It prints per-frame statuses and final totals/report paths, and returns a non-zero exit code if any sampled frame requires review or fails.

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

### Optional PostgreSQL migration integration tests

These tests are opt-in and **never** read `.env` or use the application's `POSTGRES_*` settings. They require an explicitly supplied URL whose database name is exactly `cv_integrity_migration_test`; the tests refuse any other database before connecting. Run them only against the dedicated disposable database, never against `cv_integrity` or another non-test database. Each case creates and drops a uniquely named temporary schema inside that disposable database.

Set the URL in the current PowerShell session (do not commit it or include it in logs), then run the focused test module. This prompts for the disposable database credentials instead of placing them in shell history:

```powershell
$testCredential = Get-Credential -Message "Disposable PostgreSQL test database"
$testUser = [uri]::EscapeDataString($testCredential.UserName)
$testPassword = [uri]::EscapeDataString($testCredential.GetNetworkCredential().Password)
$env:ASSURANCE_TEST_DATABASE_URL = "postgresql://${testUser}:${testPassword}@localhost:5433/cv_integrity_migration_test"
Remove-Variable testCredential, testUser, testPassword
py -3.12 -m unittest discover -s tests -p "test_database_migration_integration.py" -v
Remove-Item Env:ASSURANCE_TEST_DATABASE_URL
```

Without `ASSURANCE_TEST_DATABASE_URL`, these integration cases are clearly skipped; the regular unit suite remains offline.

## Configuration, secrets, and logging

- `config.yaml` contains project-relative paths and image/inference thresholds.
- `.env.example` gives placeholder PostgreSQL settings for localhost port `5433`. Copy it to `.env` and replace required placeholders; never commit `.env`.
- The CLI attempts to write logs to `logs/assurance.log`. If file logging cannot be configured, it falls back to standard error and continues starting.

## Current limitations

This milestone implements image and label checks, local model hash-baseline verification, CPU-based YOLO detection, image-quality heuristics, optional PostgreSQL provenance persistence, and local PDF reports. Model training and fine-tuning are not implemented. The working dataset contains five images but no valid YOLO label files, so it supports integrity and inference demonstrations but not ground-truth accuracy evaluation. Dataset contents and model weights are excluded from source control; automated tests use synthetic files and mocked inference/database connections.
