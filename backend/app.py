import os
import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timezone
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



from .database import engine, get_db, Base
from .models import (
    Patient,
    Hospital,
    AccessLog,
    MedicalReport,
    PatientMedicalFile,
    PatientFaceTemplate,
)
from .schemas import (
    PatientCreate,
    PatientResponse,
    PatientUpdate,
    HospitalCreate,
    HospitalResponse,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_hospital,
    get_current_patient,
    encrypt_face_embedding,
    decrypt_face_embedding,
)



Base.metadata.create_all(bind=engine)




FACE_MODEL_DIR = os.path.join(AI_DIR, "face_models")

FACE_DETECTOR = None
FACE_RECOGNIZER = None


def get_face_models():
    global FACE_DETECTOR, FACE_RECOGNIZER

    if FACE_DETECTOR is None or FACE_RECOGNIZER is None:
        detector_path = os.path.join(
            FACE_MODEL_DIR,
            "face_detection_yunet_2023mar.onnx"
        )
        recognizer_path = os.path.join(
            FACE_MODEL_DIR,
            "face_recognition_sface_2021dec.onnx"
        )

        detector = cv2.FaceDetectorYN_create(
            detector_path,
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        recognizer = cv2.FaceRecognizerSF_create(
            recognizer_path,
            ""
        )

        FACE_DETECTOR = detector
        FACE_RECOGNIZER = recognizer

    return FACE_DETECTOR, FACE_RECOGNIZER


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







@app.post("/patients/me/face-enrollment")
async def enroll_patient_face(
    consent_given: bool = Form(...),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_patient: dict = Depends(get_current_patient),
):
    if not consent_given:
        raise HTTPException(
            status_code=400,
            detail="Explicit consent is required for face enrollment.",
        )

    if face_image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG or PNG image.",
        )

    image_bytes = await face_image.read(5 * 1024 * 1024 + 1)
    if not image_bytes or len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be non-empty and no larger than 5 MB.",
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    face_detector, face_recognizer = get_face_models()

    height, width = image.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)

    if faces is None or len(faces) != 1:
        raise HTTPException(
            status_code=400,
            detail="The image must contain exactly one detectable face.",
        )

    aligned_face = face_recognizer.alignCrop(image, faces[0])
    embedding = face_recognizer.feature(aligned_face).flatten().tolist()

    emias_id = current_patient["emias_id"]
    encrypted_embedding = encrypt_face_embedding(embedding)

    template = (
        db.query(PatientFaceTemplate)
        .filter(PatientFaceTemplate.emias_id == emias_id)
        .first()
    )

    if template:
        template.face_embedding = encrypted_embedding
        template.consent_given = True
        template.enrolled_at = datetime.now(timezone.utc).isoformat()
    else:
        template = PatientFaceTemplate(
            emias_id=emias_id,
            face_embedding=encrypted_embedding,
            consent_given=True,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(template)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to save face enrollment.",
        )

    return {
        "message": "Face enrollment completed.",
        "emias_id": emias_id,
        "consent_given": True,
    }








@app.post("/hospital/match-patient-face")
async def match_patient_face(
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_hospital: dict = Depends(get_current_hospital),
):
    # Helper to log a face-matching outcome without storing the image.
    def log_face_attempt(action: str, emias_id: str = "UNKNOWN"):
        hospital_id = (
            current_hospital.get("hospital_id")
            or current_hospital.get("id")
        )
        hospital_email = (
            current_hospital.get("hospital_email")
            or current_hospital.get("email")
            or "unknown"
        )

        db.add(
            AccessLog(
                hospital_id=hospital_id,
                hospital_email=hospital_email,
                emias_id=emias_id,
                access_time=datetime.now(timezone.utc).isoformat(),
                action=action,
            )
        )
        db.commit()

    # Accept only JPEG and PNG images.
    if face_image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG or PNG image.",
        )

    image_bytes = await face_image.read(5 * 1024 * 1024 + 1)

    if not image_bytes or len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be non-empty and no larger than 5 MB.",
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )


    face_detector, face_recognizer = get_face_models()

    height, width = image.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image)


    if faces is None or len(faces) != 1:
        raise HTTPException(
            status_code=400,
            detail="The image must contain exactly one detectable face.",
        )


    aligned_face = face_recognizer.alignCrop(image, faces[0])
    uploaded_embedding = face_recognizer.feature(
        aligned_face
    ).flatten()


    templates = (
        db.query(PatientFaceTemplate)
        .filter(PatientFaceTemplate.consent_given.is_(True))
        .all()
    )

    best_match = None
    best_score = -1.0

    for template in templates:
        try:
            stored_embedding = np.asarray(
                decrypt_face_embedding(template.face_embedding),
                dtype=np.float32,
            )

            score = float(
                face_recognizer.match(
                    uploaded_embedding.reshape(1, -1),
                    stored_embedding.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )

            if score > best_score:
                best_score = score
                best_match = template

        except Exception:
            # Skip malformed or undecryptable templates.
            continue

    # Provisional threshold; not validated for real-world use.
    if best_match is None or best_score < 0.363:
        log_face_attempt("FACE_MATCH_NO_MATCH")

        return {
            "matched": False,
            "message": "No sufficiently similar enrolled face was found.",
        }

    patient = (
        db.query(Patient)
        .filter(Patient.emias_id == best_match.emias_id)
        .first()
    )

    if not patient:
        log_face_attempt("FACE_MATCH_PATIENT_NOT_FOUND")

        return {
            "matched": False,
            "message": "No matching patient record was found.",
        }

    log_face_attempt(
        "FACE_MATCH_POSSIBLE_MATCH",
        patient.emias_id,
    )

    return {
        "matched": True,
        "emias_id": patient.emias_id,
        "full_name": patient.full_name,
        "similarity_score": round(best_score, 4),
        "message": "Possible match found. Verify the patient's identity.",
    }







@app.put("/patients/me", response_model=PatientResponse)
def update_my_patient_record(
    patient_data: PatientUpdate,
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

    update_data = patient_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(patient, field, value)

    db.commit()
    db.refresh(patient)

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
        print("[LOGIN DEBUG] Hospital email not found in database")
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
    )

    print("[LOGIN DEBUG] Hospital email found in database")

    # Verify password
    password_valid = verify_password(
        form_data.password,
        existing_hospital.hashed_password
    )
    print(f"[LOGIN DEBUG] Password verification result: {password_valid}")

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
            from inference import predict_ecg
            from ai.gradcam import generate_gradcam
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
    
    
    

# =========================
# PATIENT MEDICAL FILES
# =========================

PATIENT_UPLOAD_DIR = os.path.join(
    REPORT_UPLOAD_DIR,
    "patient_uploads"
)

os.makedirs(PATIENT_UPLOAD_DIR, exist_ok=True)

ALLOWED_PATIENT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_PATIENT_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/patients/me/medical-files")
async def upload_patient_medical_file(
    report_type: str = Form(...),
    medical_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_patient: dict = Depends(get_current_patient),
):
    emias_id = current_patient["emias_id"]

    # Validate report type
    allowed_report_types = {
        "ECG", "X-Ray", "MRI", "CT", "Blood Report", "Other"
    }
    if report_type not in allowed_report_types:
        raise HTTPException(status_code=400, detail="Invalid report type.")

    # Validate filename extension
    original_filename = os.path.basename(medical_file.filename or "")
    extension = os.path.splitext(original_filename)[1].lower()

    if extension not in ALLOWED_PATIENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, JPG, JPEG, and PNG files are allowed."
        )

    # Read at most 10 MB + 1 byte to enforce the limit
    file_content = await medical_file.read(MAX_PATIENT_FILE_SIZE + 1)

    if not file_content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(file_content) > MAX_PATIENT_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB."
        )

    # Validate basic file signatures
    if extension == ".pdf":
        valid_file = file_content.startswith(b"%PDF-")
    elif extension in {".jpg", ".jpeg"}:
        valid_file = file_content.startswith(b"\xff\xd8\xff")
    else:
        valid_file = file_content.startswith(b"\x89PNG\r\n\x1a\n")

    if not valid_file:
        raise HTTPException(
            status_code=400,
            detail="The file content does not match its extension."
        )

    # Generate a server-controlled filename
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(PATIENT_UPLOAD_DIR, unique_filename)

    report = PatientMedicalFile(
        emias_id=emias_id,
        filename=unique_filename,
        original_filename=original_filename[:255],
        report_type=report_type,
        uploaded_at=datetime.now().isoformat(timespec="seconds"),
    )

    try:
        with open(file_path, "wb") as output_file:
            output_file.write(file_content)

        db.add(report)
        db.commit()
        db.refresh(report)

    except Exception:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail="Unable to save the medical file."
        )

    return {
        "message": "Medical file uploaded successfully.",
        "file": {
            "id": report.id,
            "filename": report.original_filename,
            "report_type": report.report_type,
            "uploaded_at": report.uploaded_at,
        },
    }


@app.get("/patients/me/medical-files")
def get_my_medical_files(
    db: Session = Depends(get_db),
    current_patient: dict = Depends(get_current_patient),
):
    reports = (
        db.query(PatientMedicalFile)
        .filter(
            PatientMedicalFile.emias_id == current_patient["emias_id"]
        )
        .order_by(PatientMedicalFile.id.desc())
        .all()
    )

    return {
        "total_files": len(reports),
        "files": [
            {
                "id": report.id,
                "filename": report.original_filename,
                "report_type": report.report_type,
                "uploaded_at": report.uploaded_at,
            }
            for report in reports
        ],
    }


@app.get("/patients/me/medical-files/{file_id}/download")
def download_my_medical_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_patient: dict = Depends(get_current_patient),
):
    report = (
        db.query(PatientMedicalFile)
        .filter(
            PatientMedicalFile.id == file_id,
            PatientMedicalFile.emias_id == current_patient["emias_id"],
        )
        .first()
    )

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Medical file not found."
        )

    file_path = os.path.join(PATIENT_UPLOAD_DIR, report.filename)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="The stored file could not be found."
        )

    extension = os.path.splitext(report.filename)[1].lower()
    media_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }

    return FileResponse(
        path=file_path,
        media_type=media_types.get(extension, "application/octet-stream"),
        filename=report.original_filename,
    )
