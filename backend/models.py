from sqlalchemy import Column, Integer, String, Text, Boolean
from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    emias_id = Column(String(20), unique=True, index=True, nullable=False)

    full_name = Column(String(100), nullable=False)
    date_of_birth = Column(String(20), nullable=False)
    gender = Column(String(20), nullable=False)

    blood_group = Column(String(10))
    phone = Column(String(20))
    address = Column(Text)

    allergies = Column(Text)
    medical_conditions = Column(Text)
    medications = Column(Text)

    emergency_contact_name = Column(String(100))
    emergency_contact_phone = Column(String(20))
    password_hash = Column(String(255), nullable=True)
    
class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)

    hospital_name = Column(String(150), nullable=False)
    registration_number = Column(String(100), unique=True, nullable=False)

    email = Column(String(150), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    role = Column(String(50), default="hospital")
    
class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)

    hospital_id = Column(Integer, nullable=True)
    hospital_email = Column(String(150), nullable=False)

    emias_id = Column(String(20), nullable=False)

    access_time = Column(String(50), nullable=False)

    action = Column(String(100), nullable=False)
    
    
    
class MedicalReport(Base):
    __tablename__ = "medical_reports"

    id = Column(Integer, primary_key=True, index=True)

    emias_id = Column(String(20), nullable=False, index=True)

    hospital_id = Column(Integer, nullable=False)

    filename = Column(String(255), nullable=False)

    report_type = Column(String(50), nullable=False)

    doctor_diagnosis = Column(Text, nullable=False)

    ai_prediction = Column(String(100), nullable=True)

    ai_confidence = Column(String(20), nullable=True)

    analysis_status = Column(String(50), nullable=True)

    uploaded_at = Column(String(50), nullable=False)
    
    uploaded_by_role = Column(String(20), nullable=False, default="hospital")
    
    
    

class PatientMedicalFile(Base):
    __tablename__ = "patient_medical_files"

    id = Column(Integer, primary_key=True, index=True)

    emias_id = Column(String(20), nullable=False, index=True)

    filename = Column(String(255), nullable=False)

    original_filename = Column(String(255), nullable=False)

    report_type = Column(String(50), nullable=False)

    uploaded_at = Column(String(50), nullable=False)





class PatientFaceTemplate(Base):
    __tablename__ = "patient_face_templates"

    id = Column(Integer, primary_key=True, index=True)

    # Links the face template to the registered EMIAS patient.
    emias_id = Column(
        String(20),
        unique=True,
        index=True,
        nullable=False
    )

    # Serialized face embedding, not the original photograph.
    face_embedding = Column(Text, nullable=False)

    # Face enrollment must be based on explicit patient consent.
    consent_given = Column(Boolean, nullable=False, default=False)

    enrolled_at = Column(String(50), nullable=False)