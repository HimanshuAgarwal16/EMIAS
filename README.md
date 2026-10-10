# EMIAS — Emergency Medical Information & Assistance System

An AI-assisted healthcare platform designed to provide secure access to essential patient medical information during emergencies and support the analysis of medical reports.

## Overview

EMIAS aims to help healthcare professionals access important patient information efficiently during emergency situations. The platform combines patient information management, hospital access, and AI-assisted medical report analysis within a secure framework.

## Key Features

- **Patient Registration:** Maintain essential patient details and medical history.
- **Unique Patient Identification:** Associate patient records with a unique identification number.
- **Hospital Module:** Provide hospital registration and controlled access to patient information.
- **Secure Access:** Support authentication and access control for sensitive medical information.
- **AI-Assisted Medical Report Analysis:** Apply machine learning and deep learning techniques to medical report analysis.
- **Explainable AI:** Use Grad-CAM to help visualize model-relevant regions and improve the interpretability of model predictions.
- **ECG Model Evaluation:** Include model evaluation and confusion matrix visualization.

## Technology Stack

- **Programming:** Python
- **AI/ML:** PyTorch, deep learning, image processing
- **Backend:** Python-based web application
- **Frontend:** HTML, CSS, JavaScript (where implemented)
- **Data Processing:** Python data loading and preprocessing tools
- **Explainability:** Grad-CAM

*Update this list to match the libraries and technologies actually used in the project.*

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

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HimanshuAgarwal16/EMIAS.git
cd EMIAS
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the application

Check `backend/app.py` for the application's entry point and required configuration. If it exposes a FastAPI application instance named `app`, you can typically start it with:

```bash
uvicorn backend.app:app --reload
```

Adjust the command if your application uses a different framework or entry point.

## Security and Privacy

EMIAS is intended to handle sensitive medical information. Use appropriate authentication, access control, secure configuration, and data protection practices. Do not commit real patient records, credentials, API keys, or confidential medical reports.

## Disclaimer

EMIAS is an academic software project. AI-generated outputs are experimental and are not a substitute for professional medical judgment or clinical validation.

## Author

**Himanshu Agarwal**

GitHub: [HimanshuAgarwal16](https://github.com/HimanshuAgarwal16)
