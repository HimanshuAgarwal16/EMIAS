from pydantic import BaseModel


class PatientCreate(BaseModel):
    full_name: str
    date_of_birth: str
    gender: str
    blood_group: str | None = None
    phone: str | None = None
    address: str | None = None
    allergies: str | None = None
    medical_conditions: str | None = None
    medications: str | None = None
    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None
    password: str

class PatientResponse(BaseModel):
    emias_id: str
    full_name: str
    date_of_birth: str
    gender: str
    blood_group: str | None = None
    phone: str | None = None
    address: str | None = None
    allergies: str | None = None
    medical_conditions: str | None = None
    medications: str | None = None
    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None

    class Config:
        from_attributes = True
        
class HospitalCreate(BaseModel):
    hospital_name: str
    registration_number: str
    email: str
    password: str


class HospitalResponse(BaseModel):
    hospital_name: str
    registration_number: str
    email: str
    role: str

    class Config:
        from_attributes = True


class HospitalLogin(BaseModel):
    email: str
    password: str