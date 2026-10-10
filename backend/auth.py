from argon2 import PasswordHasher
from jose import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer


SECRET_KEY = "EMIAS_SECRET_KEY_CHANGE_THIS_LATER"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60



oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/hospital/login"
)


password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    try:
        password_hasher.verify(
            hashed_password,
            plain_password
        )
        return True

    except Exception:
        return False


def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    
def get_current_hospital(token: str = Depends(oauth2_scheme)):

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        hospital_id = payload.get("sub")

        if hospital_id is None:
            raise credentials_exception

        return {
            "hospital_id": hospital_id,
            "email": payload.get("email"),
            "role": payload.get("role")
        }

    except Exception:
        raise credentials_exception
    
def get_current_patient(token: str = Depends(oauth2_scheme)):

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        patient_id = payload.get("sub")
        role = payload.get("role")

        if patient_id is None or role != "patient":
            raise credentials_exception

        return {
            "patient_id": patient_id,
            "emias_id": payload.get("emias_id"),
            "role": role
        }

    except Exception:
        raise credentials_exception