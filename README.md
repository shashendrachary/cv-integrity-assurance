# Integrity-Assurance

**SIH project: trustworthy computer-vision integrity assurance for data, models, and inference outputs in multi-contributor pipelines.**

Integrity-Assurance is a local Windows command-line application for reviewing image datasets, checking registered model and image files against explicit SHA-256 baselines, running CPU object detection, and producing assessment evidence. A single-image or batch assessment can record provenance in PostgreSQL when configured and generate a local PDF report. It does not provide a dashboard or establish that a baseline's original source is authentic.

## Implemented capabilities

Implemented now:

- A Python CLI entry point in `assurance.py` with a help screen and a foundation status summary.
- SHA-256 hashing of image files using bounded chunks, Pillow-based decoding checks, and byte-identical duplicate grouping.
- Explicit local SHA-256 baseline registration and verification for each trusted input image.
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


The CLI reads project identity, configured dataset paths, the model and image baseline registries, report paths, and inference/anomaly settings from `config.yaml`.

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
  image_registry: models/image_registry.json
  reports: reports
```

For example, `python assurance.py check-data --project-config projects/road/config.yaml` selects that configuration and resolves its relative resource paths from `projects/road/`. The option is supported by `check-data`, model/image registration and verification, `infer`, `assess`, `assess-batch`, `assess-video`, and `demo-model-tampering`; existing commands without it continue using the root configuration. Relative `--registry`, `--image-registry`, `--path`, `--image`, `--video`, `--images-dir`, and `--labels-dir` overrides use the selected configuration directory; absolute overrides remain absolute. The `init-db` and temporary `demo-data-integrity` commands do not use project resources and do not accept this option.

The project context validates identifiers and groups resource paths for CLI selection and attribution. It does not authenticate users or restrict filesystem access. Project IDs in provenance are attribution, not a database security boundary.

## Prerequisites and setup (PowerShell)

Use Windows with Python 3.12 (the project environment was specified for 3.12.8) and the `py` launcher. Install the requirements in a virtual environment:

```powershell
py -3.12 --version
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable in the virtual environment directly, or adjust the execution policy for your user according to your organization's guidance.

Pillow decodes images and computes image-quality metrics; OpenCV decodes video; Ultralytics supplies YOLO inference; ReportLab creates PDFs; Psycopg 3 connects to PostgreSQL. `check-data`, baseline operations, and the temporary integrity demos do not require a live PostgreSQL server. Model inference requires a locally available model file and installed Ultralytics dependencies.

## CLI usage

Run from the project root:

```powershell
py -3.12 assurance.py --help
py -3.12 assurance.py
py -3.12 assurance.py check-data
py -3.12 assurance.py register-model --id demo-model --path models/demo-model.bin
py -3.12 assurance.py verify-model --id demo-model
py -3.12 assurance.py register-image --image dataset/images/sample.jpg
py -3.12 assurance.py verify-image --image dataset/images/sample.jpg
py -3.12 assurance.py infer --id yolo-model --image dataset/images/sample.jpg
py -3.12 assurance.py assess --id yolo11n --image C:\path\to\input-image.jpg
py -3.12 assurance.py assess-batch --id yolo11n --images-dir dataset/images
py -3.12 assurance.py assess-video --id yolo11n --video dataset/videos/inspection.mp4 --interval-seconds 2
py -3.12 assurance.py demo-model-tampering --id yolo11n
py -3.12 assurance.py demo-data-integrity
py -3.12 assurance.py init-db
py -3.12 assurance.py migrate-db
```

The default command prints the project name, whether its expected directories are present, and the available CLI capabilities. `check-data` uses `paths.images` and `paths.labels` from `config.yaml`; optional `--images-dir` and `--labels-dir` arguments override either directory. It does not require PostgreSQL, credentials, model weights, or internet access.

Model registry entries are stored at `paths.model_registry` (currently `models/model_registry.json`). Relative model paths stored in the registry are resolved from the project root, regardless of the caller's current working directory; absolute model paths remain supported. Register a local model file once to record its SHA-256 baseline. Re-registering an existing ID fails unless `--replace` is explicitly supplied. Both model commands accept `--registry PATH` to override the configured registry. Registration only records a local baseline: hash agreement indicates unchanged bytes since that baseline, not authenticity of the source, accuracy, or safety. `SUSPICIOUS` means the model bytes changed and is a critical integrity failure; `REVIEW REQUIRED` means the model cannot be safely verified. The reusable `ensure_model_verified` guard blocks inference use for either state.

### Trusted input images

Image baselines are stored separately at `paths.image_registry` (currently `models/image_registry.json`) and are keyed by the image's resolved path. Register a trusted image only after reviewing its source, then verify it at any time:

```powershell
py -3.12 assurance.py register-image --image dataset/images/sample.jpg
py -3.12 assurance.py verify-image --image dataset/images/sample.jpg
```

`infer`, `assess`, `assess-batch`, and sampled video frames check this registry automatically; `--image-registry PATH` overrides it. The three image-integrity outcomes are:

| Situation | Image-integrity status | Assessment consequence |
| --- | --- | --- |
| Image matches its explicitly registered SHA-256 | `VERIFIED` | Image integrity passes. |
| Image bytes differ from the registered SHA-256 | `SUSPICIOUS` | Assessment is suspicious; detections may still be available when the model is verified. |
| No baseline exists, or baseline evidence cannot be read | `REVIEW REQUIRED` | Assessment cannot be marked verified. |

The image outcome is shown separately from model verification and object detection. Registration records the current bytes; it does not authenticate who supplied the image. The JSON registry is local project data and should itself be protected. Baselines are keyed by resolved file path, so moving or renaming an image requires registering it at its new path.

### Local YOLO inference

Download the official pretrained YOLO11 nano detection weights (`yolo11n.pt`) from the [Ultralytics assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) into the configured `models/` folder if the local file is absent. Model weights are excluded from source control; `.pt` files are ignored by Git. A first download requires internet access. With the file and dependencies available, inference runs locally on CPU.

Register the local file as a baseline, then infer:

```powershell
py -3.12 assurance.py register-model --id yolo-model --path models/yolo11n.pt
py -3.12 assurance.py infer --id yolo-model --image dataset/images/sample.jpg
```

Registration is an explicit trust decision. Inference verifies the registered model SHA-256, copies the file to a temporary private snapshot, checks the snapshot against the same baseline, then loads that snapshot only if model verification remains `VERIFIED`. Changed model weights are never auto-registered. `--conf` optionally overrides the configured confidence threshold, and inference is CPU-only by default. Detection confidence is a model score, not proof of a correct prediction.

`config.yaml` contains project/resource paths, model-verification settings, and initial image-quality heuristic thresholds for dimensions, aspect ratio, mean brightness, and grayscale Laplacian variance (blur estimate). Findings mark a run `REVIEW REQUIRED` while allowing the user to inspect the result; they are not proof of malicious activity. Thresholds need calibration against the target dataset. Invalid or unreadable images block inference.

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

`init-db` prints `PostgreSQL provenance table initialized (existing records preserved).` and creates the legacy table only if absent. Project-scoped persistence requires an intentional migration; `migrate-db` reports the schema version and states that existing provenance was preserved. New inference and assessment evidence includes image/model hashes and verification evidence, configuration, detections, image metrics, outcome, and reasons. Against a legacy schema that has not been migrated, persistence is reported as not saved while local inference and report generation remain available.

The CLI reports evidence as `PostgreSQL evidence: saved to PostgreSQL`, `skipped ...`, or `not saved ...`, as applicable, and prints a provenance run ID. These strings describe the outcome of that run; they are not a database dump or proof of a particular deployment. The automated assessment tests exercise provenance insertion with a mock inserter. Separately, a manual assessment was verified to save its provenance record to the configured local PostgreSQL database, and a read-only query confirmed that the corresponding run ID existed in `inference_provenance`. This demonstrates a successful local persistence check, not a claim about any other deployment.

Provenance is a record of data observed by this application, not proof that a model source is authentic or that a prediction is objectively correct. Database availability does not bypass model verification.

### End-to-end assessment and PDF

Run a complete integrity-gated assessment using the existing registered model and an image file supplied locally:

```powershell
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python assurance.py assess --id yolo11n --image C:\path\to\input-image.jpg
```

Each completed assessment writes a uniquely named PDF under `paths.reports` (currently `reports/`) and prints `PDF report: <generated path>`. The CLI also prints `Overall status`, `Provenance run ID`, and `PostgreSQL evidence`. A PDF is still generated when persistence is skipped or fails, provided local assessment evidence can be generated. The assessment tests observed a generated PDF beginning with the PDF file signature and a report path; the recorded database insertion in those tests used a mock.

Assessment status reflects image/model verification, image-quality findings, and whether inference succeeded. A trusted, unchanged image reports `VERIFIED` image integrity; changed bytes report `SUSPICIOUS`; a missing baseline reports `REVIEW REQUIRED`. Model mismatch is also `SUSPICIOUS` and blocks loading that model. Missing model evidence, image anomalies, or inference failure require review. PostgreSQL persistence is reported separately and does not turn local model/image verification into a different integrity result. A matching hash shows only that bytes match a locally stored baseline. These rule-based results do not prove original authenticity, safety, accuracy, or prediction correctness. Anomaly thresholds are initial heuristics and require dataset-specific calibration.

### Batch image assessment

Assess every supported image beneath a directory, recursively and in deterministic path order:

```powershell
python assurance.py assess-batch --id yolo11n --images-dir dataset/images
```

The batch command reuses the single-image assessment workflow, including the image baseline registry, and does not create per-image PDFs. After all images have been attempted, it writes one consolidated PDF and a matching JSON sidecar containing detection details. The PDF summarizes discovered and processed images, successful/review-required/failed outcomes, model verification, image hashes and properties, quality flags, aggregate detections, and findings. It prints each image's status and the final report paths, then returns a non-zero exit code if any image requires review or fails. `--images-dir` is required; relative paths use the selected project configuration directory. The command also accepts `--conf`, `--registry`, `--image-registry`, and `--project-config`.

### Video frame assessment

Assess frames sampled once per second by default:

```powershell
python assurance.py assess-video --id yolo11n --video dataset/videos/inspection.mp4 --interval-seconds 2.0
```

Video decoding uses OpenCV. The command processes frames sequentially without loading the entire video into memory, and each sampled frame runs through model verification and the assessment workflow. It generates one consolidated PDF and JSON sidecar with frame-level detection details. The report/provenance identifies the video path and records the sampled frame index, time, interval, and frame-hash scope; temporary frame files are removed after assessment. The command does not connect to PostgreSQL. Because the sampled frame image is temporary and its path changes on each run, it has no registered image baseline by default and receives `REVIEW REQUIRED`; this is the current video-frame baseline caveat. It accepts `--conf`, `--registry`, `--image-registry`, and `--project-config`; the interval must be finite and greater than zero. It prints per-frame statuses and final totals/report paths, and returns a non-zero exit code if any sampled frame requires review or fails.

## Demo sequence

This checkout contains the `models/yolo11n.pt` model weights and the sample image `dataset/images/busy_street.jpg`, along with previously generated assessment reports. Video availability depends on the local dataset. From the project root, use a disposable image copy and a temporary image registry for the image-baseline demonstration. Supply your own supported video path if demonstrating video assessment:

```powershell
$demoRoot = Join-Path $env:TEMP "integrity-assurance-demo"
New-Item -ItemType Directory -Force -Path $demoRoot
$demoImage = Join-Path $demoRoot "demo-image.jpg"
$demoRegistry = Join-Path $demoRoot "image_registry.json"
Copy-Item C:\path\to\source-image.jpg $demoImage
python assurance.py verify-model --id yolo11n
python assurance.py register-image --image $demoImage --registry $demoRegistry
python assurance.py verify-image --image $demoImage --registry $demoRegistry
python assurance.py assess --id yolo11n --image $demoImage --image-registry $demoRegistry
python assurance.py verify-image --image C:\path\to\unregistered-image.jpg --registry $demoRegistry
Get-ChildItem reports\assurance-*.pdf | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python assurance.py assess-batch --id yolo11n --images-dir C:\path\to\demo-images
python assurance.py assess-video --id yolo11n --video C:\path\to\demo-video.mp4 --interval-seconds 2
python assurance.py demo-model-tampering --id yolo11n
python assurance.py demo-data-integrity
```

For the `SUSPICIOUS` outcome, replace the copied demo image with a different valid image using an image editor, then run `verify-image` again with the same temporary registry. The unregistered-image command demonstrates `REVIEW REQUIRED`; the registered unchanged copy demonstrates `VERIFIED`. The existing integrity unit tests observed all three outcomes using synthetic temporary images. Do not change the source image.

`demo-model-tampering` requires an intact registered source model. It copies the file into a temporary directory, creates a temporary baseline, modifies only that copy, and shows the expected/actual hashes and `SUSPICIOUS` status. It then uses a mocked loader and confirms inference is blocked before loading. The temporary directory is removed and the source hash is checked for change. `demo-data-integrity` creates valid/corrupt/duplicate images, a malformed label, and an image missing its label in a temporary directory; it prints the findings and cleans up all demo files. Neither demo writes to the real dataset or model files.

Image integrity outcome meanings are `VERIFIED` for a trusted unchanged image, `SUSPICIOUS` for a hash mismatch, and `REVIEW REQUIRED` when no usable image baseline exists. A missing image baseline does not block model loading or object detection, but prevents a successful overall assessment. A model hash mismatch is `SUSPICIOUS` and blocks loading. Overall status also reflects image-quality findings and inference success; PostgreSQL persistence has its own printed status. Hash agreement means unchanged bytes since local registration, not original authenticity or prediction correctness.

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

This milestone implements image and label checks, local image and model hash-baseline verification, CPU-based YOLO detection, image-quality heuristics, optional PostgreSQL provenance persistence, and local PDF reports. Model training and fine-tuning are not implemented. The checked-out `dataset/images/` directory includes the sample image `busy_street.jpg`; additional image and video assets may need to be supplied locally for image, batch, and video demonstrations. The dataset labels must be valid and available for ground-truth accuracy evaluation; integrity checks and detections do not measure accuracy. Image baselines are local JSON records keyed by resolved path, so registry loss, untrusted registry edits, moved images, or unreviewed registrations affect the trust decision. Dataset contents and model weights are excluded from source control; automated tests use synthetic files and mocked inference/database connections. The automated assessment tests use mocked persistence. Separately, a manual assessment was verified to save its provenance record to the configured local PostgreSQL database, and a read-only query confirmed that the corresponding run ID existed in `inference_provenance`. This demonstrates a successful local persistence check, not a claim about any other deployment.
