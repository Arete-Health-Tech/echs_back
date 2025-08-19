from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pymongo
import bcrypt
from jose import jwt, JWTError
from datetime import datetime, timedelta
from bson import ObjectId
import openai
import base64
import json
import os
from dotenv import load_dotenv
from gridfs import GridFS
from typing import Optional

load_dotenv()

# MongoDB connection
client = pymongo.MongoClient(
    os.getenv("MONGO_URL")
)
db = client["hospital_app"]
users_collection = db["users"]

# JWT config
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Helpers
def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = users_collection.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Registration
@app.post("/register")
def register(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    age: int = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user_data = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "age": age,
        "password": hashed_pw
    }
    result = users_collection.insert_one(user_data)
    return {"message": "Registered successfully", "user_id": str(result.inserted_id)}

# Login
@app.post("/login")
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    token = create_access_token({"sub": user["email"]})

    user_data = {
        "user_id": str(user["_id"]),
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "full_name": f"{user['first_name']} {user['last_name']}",
        "email": user["email"],
        "phone": user["phone"],
        "age": user["age"]
    }

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }


# Profile
@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "full_name": f"{current_user['first_name']} {current_user['last_name']}",
        "email": current_user["email"],
        "phone": current_user["phone"],
        "age": current_user["age"]
    }

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API")

# New collection for OCR results
ocr_collection = db["ocr_results"]
file_collection = db["files"]

fs = GridFS(db)

# ---------- 1. ECHS Card/Temporary Slip Extraction ----------
@app.post("/extract/echs_card")
async def extract_echs_card(
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Read file
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Upload file to GridFS
        file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        
        # OCR Prompt
        prompt = """
        You are analyzing an ECHS Card. Extract the following fields exactly as seen:
        - Card No
        - Patient Name
        - ESM
        - DOB
        - Relationship with ESM
        There is no separate field called "Patient Name", but it is written on top left.
        If any field is missing, return "Not Found".
        Return only valid JSON.
        """

        data = await run_ocr_prompt(prompt, base64_image)

        result = ocr_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "doc_type": "echs_card",
            "image_file_id": file_id,
            "extracted_data": data,
            "uploaded_at": datetime.utcnow()
        })

        return {
            "status": "success", 
            "doc_type": "echs_card", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/temporary_slip")
async def extract_temporary_slip(
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Read file
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Upload file to GridFS
        file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        
        # OCR Prompt
        prompt = """
        You are analyzing a temporary slip, which is given if ECHS Card is not present. Extract the following fields exactly as seen:
        - Form No
        - Temporary Slip No
        - Patient Name
        - ESM
        - DOB
        - Relationship with ESM
        - Category
        - Valid Upto
        - Category of Ward
        - OIC Stamp
        There is no separate field called "Patient Name".
        If any field is missing, return "Not Found".
        Return only valid JSON.
        """

        data = await run_ocr_prompt(prompt, base64_image)

        result = ocr_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "doc_type": "temporary_slip",
            "image_file_id": file_id,
            "extracted_data": data,
            "uploaded_at": datetime.utcnow()
        })

        return {
            "status": "success", 
            "doc_type": "temporary_slip", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 2. Referral Letter Extraction ----------
@app.post("/extract/referral_letter")
async def extract_referral_letter(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Upload file to GridFS
        file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)

        prompt = """
        You are analyzing a Referral Letter. Extract:
        - Name of Patient
        - Referral No
        - Valid Upto
        - Date of Issue
        - No of Sessions Allowed
        - Patient Type
        - Age
        - Gender
        - Relationship with ESM
        - Category
        - Service No
        - Card No
        - ESM Name
        - ESM Contact Number
        - Clinical Notes
        - Admission
        - Investigation
        - Consultation For
        - Polyclinic Remarks
        - Claim ID
        If missing, return "Not Found".
        Return only valid JSON.
        """

        data = await run_ocr_prompt(prompt, base64_image)

        result = ocr_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "doc_type": "referral_letter",
            "image_file_id": file_id,
            "extracted_data": data,
            "uploaded_at": datetime.utcnow()
        })

        return {
            "status": "success", 
            "doc_type": "referral_letter",
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id), 
            "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 3. Aadhar Card Extraction ----------
@app.post("/extract/aadhar_card")
async def extract_aadhar_card(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Upload file to GridFS
        file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)

        prompt = """
        You are analyzing an Aadhaar Card. Extract:
        - Aadhaar No
        - Name
        - Date of Birth
        - Gender
        Do not preassume anything.
        If missing, return "Not Found".
        Return only valid JSON.
        """

        data = await run_ocr_prompt(prompt, base64_image)

        result = ocr_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "doc_type": "aadhar_card",
            "image_file_id": file_id,
            "extracted_data": data,
            "uploaded_at": datetime.utcnow()
        })

        return {
            "status": "success", 
            "doc_type": "aadhar_card", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Helper Function for OCR ----------
async def run_ocr_prompt(prompt, base64_image):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\nReturn ONLY valid JSON, no explanations."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=1500
    )

    reply = response['choices'][0]['message']['content'].strip()
    if reply.startswith("```json"):
        reply = reply.removeprefix("```json").removesuffix("```").strip()
    elif reply.startswith("```"):
        reply = reply.removeprefix("```").removesuffix("```").strip()

    return json.loads(reply)

# Retrieving the stored image
@app.get("/image/{file_id}")
def get_image(file_id: str, current_user: dict = Depends(get_current_user)):
    grid_out = fs.get(ObjectId(file_id))
    return StreamingResponse(grid_out, media_type=grid_out.content_type)

# Collection for storing history
requests_collection = db["requests_history"]

class FinalSubmissionRequest(BaseModel):
    echs_card_result_id: Optional[str] = None
    referral_letter_result_id: Optional[str] = None
    aadhar_card_result_id: Optional[str] = None
    
class SubmitRequestPayload(BaseModel):
    matched: Optional[bool] = None

# Submit request API (stores OCR doc IDs, not file_ids)
@app.post("/submit_request")
def submit_request(payload: SubmitRequestPayload, current_user: dict = Depends(get_current_user)):
    echs_or_slip = ocr_collection.find_one(
        {"user_id": str(current_user["_id"]), "doc_type": {"$in": ["echs_card", "temporary_slip"]}},
        sort=[("_id", -1)]
    )
    referral = ocr_collection.find_one(
        {"user_id": str(current_user["_id"]), "doc_type": "referral_letter"},
        sort=[("_id", -1)]
    )
    aadhar = ocr_collection.find_one(
        {"user_id": str(current_user["_id"]), "doc_type": "aadhar_card"},
        sort=[("_id", -1)]
    )

    if not (echs_or_slip or referral or aadhar):
        raise HTTPException(status_code=400, detail="No documents found for submission")

    request_doc = {
        "user_id": str(current_user["_id"]),
        "echs_card_result_id": str(echs_or_slip["_id"]) if echs_or_slip else None,
        "referral_letter_result_id": str(referral["_id"]) if referral else None,
        "aadhar_card_result_id": str(aadhar["_id"]) if aadhar else None,
        "matched": payload.matched,
        "created_at": datetime.utcnow()
    }

    result = requests_collection.insert_one(request_doc)

    # Add request_id to request_doc and convert all ObjectIds to str for response
    request_doc["_id"] = str(result.inserted_id)

    return {"status": "success", "request_id": request_doc["_id"], "request_doc": request_doc}

# Get history for logged-in user
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    history = list(requests_collection.find({"user_id": str(current_user["_id"])}))
    for h in history:
        h["_id"] = str(h["_id"])
    return {"status": "success", "history": history}

@app.get("/admin/user_history/{user_id}")
def get_user_history(user_id: str):
    try:
        history = list(requests_collection.find({"user_id": user_id}))
        for h in history:
            h["_id"] = str(h["_id"])

            def attach_ocr_data(field_name, key_name):
                if h.get(field_name):
                    ocr_result = ocr_collection.find_one({"_id": ObjectId(h[field_name])})
                    if ocr_result:
                        h[key_name] = {
                            "id": str(ocr_result["_id"]),
                            "data": ocr_result.get("extracted_data", {}),
                            "uploaded_at": ocr_result.get("uploaded_at")
                        }
                    else:
                        h[key_name] = None
                else:
                    h[key_name] = None
                h.pop(field_name, None)  

            attach_ocr_data("echs_card_result_id", "echs_card_or_temporary_slip")
            attach_ocr_data("referral_letter_result_id", "referral_letter")
            attach_ocr_data("aadhar_card_result_id", "aadhar_card")

        return {
            "status": "success",
            "user_id": user_id,
            "submission_count": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/user_stats")
def get_user_stats():
    try:
        pipeline = [
            {"$group": {"_id": "$user_id", "submission_count": {"$sum": 1}}}
        ]
        stats = list(requests_collection.aggregate(pipeline))
        # Attach user info
        for stat in stats:
            user = users_collection.find_one({"_id": ObjectId(stat["_id"])})
            stat["user"] = {
                "user_id": stat["_id"],
                "full_name": f"{user['first_name']} {user['last_name']}",
                "email": user["email"]
            }
            stat["_id"] = str(stat["_id"]) 
        return {"status": "success", "user_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
