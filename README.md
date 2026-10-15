# EMIAS — Emergency Medical Information & Assistance System

An AI-assisted healthcare platform designed to provide secure access to essential patient medical information during emergencies and support medical-report analysis.

## Overview

EMIAS is an academic healthcare software project that combines patient medical information management, hospital access, face recognition, and AI-assisted report analysis.

The system aims to help authorized healthcare professionals retrieve important patient information efficiently during emergency situations. It uses authentication, role-based access controls, and audit logging to support controlled access to sensitive medical records.

## Key Features

### Patient Module
- Patient registration and authentication.
- Unique EMIAS identification for patient records.
- Medical profile viewing and updating.
- Medical-file upload and history.
- Consent-based face enrollment.

### Hospital Module
- Hospital registration and authentication.
- Authenticated access to patient records using an EMIAS ID.
- Face-based patient matching assistance.
- Access-log viewing and tracking of patient-record access.

### AI and Medical-Report Analysis
- Deep-learning-based ECG analysis.
- Model training, inference, and evaluation utilities.
- Grad-CAM visualizations to help interpret model predictions.
- Medical-report upload, analysis results, and report history.

### Security Features
- Argon2 password hashing.
- JWT-based authentication.
- Patient and hospital role validation.
- Encrypted face embeddings using Fernet.
- Audit logging for patient-record access and face-matching attempts.
- Consent checks for face enrollment.

## Technology Stack

- **Programming Language:** Python
- **Backend Framework:** FastAPI
- **Database:** SQLite
- **ORM:** SQLAlchemy
- **Authentication:** JWT and OAuth2 bearer authentication
- **Password Security:** Argon2
- **Encryption:** Cryptography (Fernet)
- **AI and Deep Learning:** PyTorch and Torchvision
- **Computer Vision:** OpenCV
- **Explainable AI:** Grad-CAM
- **Frontend:** HTML, CSS, and JavaScript

## Project Structure

```text
EMIAS/
├── ai/
│   ├── data_loader.py
│   ├── dataset.py
│   ├── ecg_model.py
│   ├── evaluate_model.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── train_model.py
├── backend/
│   ├── app.py
│   ├── auth.py
│   ├── crud.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/
├── .gitignore
└── README.md
```

The project also uses face-recognition model files under `ai/face_models/`. Other datasets, trained weights, and supporting files may be present in the local project.

## Getting Started

### Prerequisites

- Python installed on your system.
- Git.
- A Windows PowerShell terminal or another compatible terminal.
- The project dependencies and required model files.

### 1. Clone the repository

```bash
git clone https://github.com/HimanshuAgarwal16/EMIAS.git
cd EMIAS
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

If PowerShell blocks virtual-environment activation, use the appropriate execution policy permitted on your system or activate the environment through your IDE.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

The backend requires these environment variables:

- `EMIAS_SECRET_KEY` — used for JWT signing.
- `EMIAS_FACE_ENCRYPTION_KEY` — used to encrypt and decrypt face embeddings.

Generate strong keys securely, keep them private, and configure them in your local environment before starting the application. Keep the same face-encryption key for existing encrypted face templates. Do not commit either key to GitHub.

### 5. Start the backend

From the project root:

```powershell
uvicorn backend.app:app --reload
```

Once the backend starts, open the interactive API documentation:

http://127.0.0.1:8000/docs

Use the frontend files and existing application workflow to access the patient and hospital interfaces.

**Note:** The exact frontend launch method depends on the current frontend setup. Follow the application's existing configuration if it requires a separate frontend server.

## Testing and Verification

The following workflows have been exercised during development:

- Patient login and profile updates.
- Hospital login and patient-record retrieval.
- Rejection of unauthenticated patient-record requests.
- Hospital-role validation against patient tokens.
- Patient-role validation against hospital-only endpoints.
- Consent-based face enrollment.
- Possible-match and no-match face-recognition scenarios.
- Medical-report upload, AI result generation, and report history.
- Audit logging for patient-record access and face-matching attempts.

These checks cover the tested scenarios; they do not constitute a complete security audit or clinical validation.

## Security and Privacy

EMIAS is an academic project that handles sensitive medical information. Use synthetic test data during development and demonstrations.

- Never commit credentials, access tokens, encryption keys, or confidential medical records.
- Restrict access to patient records through authentication and authorization.
- Keep encryption keys secure and stable.
- Obtain consent before enrolling face data.
- Treat face-recognition results as possible matches requiring independent verification.
- Protect uploaded medical files and review access permissions before deployment.

Additional security, privacy, usability, and clinical validation are necessary before any real-world healthcare deployment.

## Limitations and Future Scope

Potential future improvements include:

- More comprehensive automated tests.
- Expanded security and authorization testing.
- Improved privacy controls and consent withdrawal.
- Evaluation of face-recognition accuracy and spoof resistance.
- Further validation of AI predictions on appropriate datasets.
- Enhanced monitoring, deployment configuration, and backup mechanisms.
- Integration with applicable digital-health systems where feasible and authorized.

## Disclaimer

EMIAS is an academic software project. Its AI-generated outputs are experimental and are not a substitute for professional medical judgment, clinical diagnosis, or emergency medical care. The system has not been established as clinically validated for real-world patient care.

## Author

**Himanshu Agarwal**

GitHub: https://github.com/HimanshuAgarwal16
