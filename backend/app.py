import os
from fastapi import FastAPI, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import sys

AI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ai"
)

if AI_DIR not in sys.path:
    sys.path.append(AI_DIR)

from inference import predict_ecg
from ai.gradcam import generate_gradcam

from .database import engine, get_db, Base
from .models import Patient, Hospital, AccessLog, MedicalReport
from .schemas import (
    PatientCreate,
    PatientResponse,
    HospitalCreate,
    HospitalResponse,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_hospital,
    get_current_patient
)



Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="EMIAS",
    description="Emergency Medical Information & Assistance System",
    version="1.0.0"
)


# =========================
# MEDICAL REPORT STORAGE
# =========================

REPORT_UPLOAD_DIR = "medical_reports"

GRADCAM_OUTPUT_DIR = os.path.join(
    REPORT_UPLOAD_DIR,
    "gradcam"
)

os.makedirs(
    REPORT_UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_OUTPUT_DIR,
    exist_ok=True
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return FileResponse(
        FRONTEND_DIR / "home.html"
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# PATIENT REGISTRATION
# ---------------------------------------------------------

@app.post("/patients", response_model=PatientResponse)
def register_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db)
):
    emias_id = "EMIAS-" + uuid.uuid4().hex[:8].upper()

    patient_data = patient.model_dump()

    password = patient_data.pop("password")

    hashed_password = hash_password(password)

    new_patient = Patient(
        emias_id=emias_id,
        password_hash=hashed_password,
        **patient_data
    )

    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return new_patient



@app.get("/patients/me", response_model=PatientResponse)
def get_my_patient_record(
    db: Session = Depends(get_db),
    current_patient: dict = Depends(get_current_patient)
):
    patient = (
        db.query(Patient)
        .filter(
            Patient.emias_id == current_patient["emias_id"]
        )
        .first()
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    return patient
# ---------------------------------------------------------
# GET PATIENT - PROTECTED BY JWT
# ---------------------------------------------------------

@app.get("/patients/{emias_id}", response_model=PatientResponse)
def get_patient(
    emias_id: str,
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital)
):
    patient = (
        db.query(Patient)
        .filter(Patient.emias_id == emias_id)
        .first()
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    # Create access log
    access_log = AccessLog(
        hospital_id=int(current_hospital["hospital_id"]),
        hospital_email=current_hospital["email"],
        emias_id=patient.emias_id,
        access_time=datetime.utcnow().isoformat(),
        action="VIEW_PATIENT_RECORD"
    )

    db.add(access_log)
    db.commit()

    return patient



@app.get("/patients/lookup/{emias_id}", response_model=PatientResponse)
def lookup_patient(
    emias_id: str,
    db: Session = Depends(get_db)
):
    patient = (
        db.query(Patient)
        .filter(Patient.emias_id == emias_id)
        .first()
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    return patient



# ---------------------------------------------------------
# HOSPITAL REGISTRATION
# ---------------------------------------------------------

@app.post("/auth/hospital/register", response_model=HospitalResponse)
def register_hospital(
    hospital: HospitalCreate,
    db: Session = Depends(get_db)
):
    # Check if email already exists
    existing_email = (
        db.query(Hospital)
        .filter(Hospital.email == hospital.email)
        .first()
    )

    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Hospital email already registered"
        )

    # Check registration number
    existing_registration = (
        db.query(Hospital)
        .filter(
            Hospital.registration_number
            == hospital.registration_number
        )
        .first()
    )

    if existing_registration:
        raise HTTPException(
            status_code=400,
            detail="Hospital registration number already exists"
        )

    # Hash password
    hashed_password = hash_password(
        hospital.password
    )

    # Create hospital
    new_hospital = Hospital(
        hospital_name=hospital.hospital_name,
        registration_number=hospital.registration_number,
        email=hospital.email,
        hashed_password=hashed_password,
        role="hospital"
    )

    db.add(new_hospital)
    db.commit()
    db.refresh(new_hospital)

    return new_hospital


# ---------------------------------------------------------
# HOSPITAL LOGIN
# ---------------------------------------------------------

@app.post("/auth/hospital/login")
def hospital_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Find hospital using email
    existing_hospital = (
        db.query(Hospital)
        .filter(Hospital.email == form_data.username)
        .first()
    )

    if not existing_hospital:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Verify password
    password_valid = verify_password(
        form_data.password,
        existing_hospital.hashed_password
    )

    if not password_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Create JWT token
    access_token = create_access_token(
        data={
            "sub": str(existing_hospital.id),
            "email": existing_hospital.email,
            "role": existing_hospital.role
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
    
    
@app.post("/auth/patient/login")
def patient_login(
    emias_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):

    patient = (
        db.query(Patient)
        .filter(Patient.emias_id == emias_id)
        .first()
    )

    if not patient:
        raise HTTPException(
            status_code=401,
            detail="Invalid EMIAS ID or password"
        )

    if not patient.password_hash:
        raise HTTPException(
            status_code=401,
            detail="Patient password is not set"
        )

    if not verify_password(
        password,
        patient.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid EMIAS ID or password"
        )

    access_token = create_access_token({
        "sub": str(patient.id),
        "emias_id": patient.emias_id,
        "role": "patient"
    })

    return {
        "message": "Patient login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "emias_id": patient.emias_id
    }
    
@app.get("/hospital/access-logs")
def get_access_logs(
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital)
):
    logs = (
        db.query(AccessLog)
        .filter(
            AccessLog.hospital_id
            == int(current_hospital["hospital_id"])
        )
        .all()
    )

    return logs

# ---------------------------------------------------------
# AI MEDICAL REPORT UPLOAD
# ---------------------------------------------------------

@app.post("/ai/upload-report")
async def upload_medical_report(
    emias_id: str = Form(...),
    report_type: str = Form(...),
    doctor_diagnosis: str = Form(...),
    medical_report: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital)
):
    
        # -------------------------
    # VERIFY PATIENT
    # -------------------------

    patient = (
        db.query(Patient)
        .filter(Patient.emias_id == emias_id)
        .first()
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient with this EMIAS ID was not found."
        )

    # Allowed report types
    allowed_report_types = {
        "ECG",
        "X-Ray",
        "MRI",
        "CT",
        "Blood Report",
        "Other"
    }

    if report_type not in allowed_report_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid medical report type."
        )

    # Allowed file extensions
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".pdf"
    }

    filename = medical_report.filename or ""

    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: JPG, JPEG, PNG, PDF."
        )

    # Generate unique filename
    unique_filename = (
        f"{emias_id}_{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        REPORT_UPLOAD_DIR,
        unique_filename
    )

    # Read uploaded file
    file_content = await medical_report.read()

    # Save uploaded report
    with open(file_path, "wb") as file:
        file.write(file_content)

    # -------------------------
    # AI ANALYSIS
    # -------------------------

    ai_result = None

    if report_type == "ECG":

        try:
            ai_result = predict_ecg(
                file_path
            )

            # Generate Grad-CAM explanation
            gradcam_filename = (
                f"{os.path.splitext(unique_filename)[0]}_gradcam.png"
            )

            gradcam_path = os.path.join(
                GRADCAM_OUTPUT_DIR,
                gradcam_filename
            )

            generate_gradcam(
                file_path,
                gradcam_path
            )

            ai_result["gradcam_filename"] = gradcam_filename

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"AI analysis failed: {str(e)}"
            )

    # -------------------------
    # COMPARE AI WITH DOCTOR
    # -------------------------

    analysis_status = "Not Available"

    if ai_result:

        doctor_text = doctor_diagnosis.lower().strip()
        ai_text = ai_result["prediction"].lower().strip()


        def normalize_diagnosis(text: str) -> str:
            text = text.lower().strip()

            if (
                "history of myocardial infarction" in text
                or "history of mi" in text
                or "old myocardial infarction" in text
            ):
                return "History of Myocardial Infarction"

            if (
                "normal ecg" in text
                or "normal ekg" in text
                or text == "normal"
            ):
                return "Normal"

            if (
                "abnormal heartbeat" in text
                or "arrhythmia" in text
                or "irregular heartbeat" in text
            ):
                return "Abnormal Heartbeat"

            if (
                "myocardial infarction" in text
                or text == "mi"
                or "heart attack" in text
            ):
                return "Myocardial Infarction"

            return text


        normalized_doctor = normalize_diagnosis(
            doctor_text
        )

        normalized_ai = normalize_diagnosis(
            ai_text
        )

        if normalized_doctor == normalized_ai:  
            analysis_status = "Consistent"
        else:
            analysis_status = "Mismatch"

    # -------------------------
    # SAVE REPORT IN DATABASE
    # -------------------------

    report_record = MedicalReport(
        emias_id=emias_id,
        hospital_id=int(
            current_hospital["hospital_id"]
        ),
        filename=unique_filename,
        report_type=report_type,
        doctor_diagnosis=doctor_diagnosis,
        ai_prediction=(
            ai_result["prediction"]
            if ai_result
            else None
        ),
        ai_confidence=(
            str(ai_result["confidence"])
            if ai_result
            else None
        ),
        analysis_status=analysis_status,
        uploaded_at=datetime.utcnow().isoformat()
    )

    db.add(report_record)
    db.commit()
    db.refresh(report_record)

    # -------------------------
    # API RESPONSE
    # -------------------------

    return {
        "status": "success",
        "message": "Medical report uploaded successfully.",
        "emias_id": emias_id,
        "report_type": report_type,
        "doctor_diagnosis": doctor_diagnosis,
        "filename": unique_filename,
        "hospital_id": current_hospital["hospital_id"],
        "ai_analysis": ai_result,
        "analysis_status": analysis_status
    }
    
    

# ---------------------------------------------------------
# SECURE GRAD-CAM IMAGE ACCESS
# ---------------------------------------------------------

@app.get("/ai/gradcam/{filename}")
def get_gradcam(
    filename: str,
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital)
):
    # Prevent path traversal
    if os.path.basename(filename) != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    gradcam_path = os.path.join(
        GRADCAM_OUTPUT_DIR,
        filename
    )

    if not os.path.isfile(gradcam_path):
        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image not found."
        )

    # Find the original report belonging to this hospital
    gradcam_stem = filename.replace(
        "_gradcam.png",
        ""
    )

    reports = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.hospital_id
            == int(current_hospital["hospital_id"])
        )
        .all()
    )

    report_exists = any(
        os.path.splitext(report.filename)[0]
        == gradcam_stem
        for report in reports
    )

    if not report_exists:
        raise HTTPException(
            status_code=403,
            detail="You are not authorized to access this Grad-CAM image."
        )

    return FileResponse(
        gradcam_path,
        media_type="image/png"
    )
    
@app.get("/ai/report-history/{emias_id}")
def get_report_history(
    emias_id: str,
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital)
):
    reports = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.emias_id == emias_id,
            MedicalReport.hospital_id
            == int(current_hospital["hospital_id"])
        )
        .order_by(MedicalReport.id.desc())
        .all()
    )

    return {
        "emias_id": emias_id,
        "total_reports": len(reports),
        "reports": [
            {
                "id": report.id,
                "filename": report.filename,
                "report_type": report.report_type,
                "doctor_diagnosis": report.doctor_diagnosis,
                "ai_prediction": report.ai_prediction,
                "ai_confidence": report.ai_confidence,
                "analysis_status": report.analysis_status,
                "uploaded_at": report.uploaded_at
            }
            for report in reports
        ]
    }