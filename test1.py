import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
from openai import OpenAI
import base64
import json
import os
import pandas as pd
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from gridfs import GridFS
from typing import Optional
import re
from playwright.sync_api import sync_playwright
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB connection
client_mongo = pymongo.MongoClient(
    os.getenv("MONGO_URL")
)
db = client_mongo[os.getenv("MONGO_NAME")]
users_collection = db["users"]
ocr_collection = db["ocr_results"]
file_collection = db["files"]
requests_collection = db["requests_history"]

# JWT config
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

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

# OpenAI client (new SDK)
client = OpenAI(api_key=os.getenv("OPENAI_API"))
fs = GridFS(db)

# JWT Helpers
def create_access_token(data: dict):
    to_encode = data.copy()
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
        logger.info("Registration attempt with existing email")
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

# Logout
@app.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    return {"status": "success", "message": "Logged out successfully"}

# Profile
@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "full_name": f"{current_user['first_name']} {current_user['last_name']}",
        "email": current_user["email"],
        "phone": current_user["phone"],
        "age": current_user["age"]
    }

# Debug endpoint to check database connection
@app.get("/debug/db_connection")
def check_db_connection():
    try:
        # Test connection
        client_mongo.admin.command('ismaster')
        doc_count = ocr_collection.count_documents({})
        return {"status": "connected", "ocr_documents": doc_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------- Helper Function for OCR with Enhanced Error Handling ----------
async def run_ocr_prompt(prompt, base64_image):
    try:
        logger.info("Starting OCR processing")
        response = client.chat.completions.create(
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
            max_tokens=3000
        )

        reply = response.choices[0].message.content.strip()
        logger.info(f"OCR response received: {reply[:100]}...")

        # Clean up response
        if reply.startswith("```
            reply = reply.removeprefix("```json").removesuffix("```
        elif reply.startswith("```"):
            reply = reply.removeprefix("``````").strip()

        # Validate JSON
        try:
            data = json.loads(reply)
            logger.info("OCR JSON parsed successfully")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise Exception(f"Invalid JSON response from OCR: {str(e)}")
            
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        raise Exception(f"OCR processing error: {str(e)}")

# ---------- 1. ECHS Card/Temporary Slip Extraction with Enhanced Error Handling ----------
@app.post("/extract/echs_card")
async def extract_echs_card(
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"ECHS card extraction started for user: {current_user['email']}")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file provided")

        base64_image = base64.b64encode(contents).decode("utf-8")

        # Upload file to GridFS
        try:
            file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
            logger.info(f"File uploaded to GridFS with ID: {file_id}")
        except Exception as e:
            logger.error(f"GridFS upload error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        
        # OCR Prompt
        prompt = """You are extracting structured data from an ECHS smart card image. Read all printed and handwritten text and return one JSON object that strictly follows the schema and rules below. No extra keys, comments, or prose. If a value is missing or illegible, output exactly 'Not Found'. Return only JSON.
                Schema (key order must be preserved):
                {
                "Card No": string, // Printed card/reg number on the card face (e.g., 'JB 0000 0268 6390'); preserve spaces exactly.
                "Patient Name": string, // Name at the top-left header; transcribe exactly as printed.
                "ESM": string, // ESM field is the printed name of the ex-serviceman only; include rank and name exactly as shown (do not include service number here).
                "Relationship with ESM": string, // Relationship term on the card such as 'Spouse', 'Son', 'Daughter', etc.; if absent, 'Not Found'.
                "DOB": string, // Value labeled 'DOB'; normalize to DD MMM YYYY when month is textual; otherwise keep as printed.
                "DOM": string, // Value labeled 'DOM'; same normalization rule; if absent, 'Not Found'.
                "Service No": string // Service number of the ESM: prefer a printed value labeled 'Service No'; if not present, use a clear handwritten alphanumeric near the name/photo area that matches common service-number patterns (e.g., 'JC257424Y', 'IC 12345', 'SS-12345'); preserve case, spaces, and hyphens exactly. If both printed and handwritten exist, use the printed one. If none, 'Not Found'.
                }

                Extraction rules:

                Treat 'ESM' strictly as the name line of the ex-serviceman (rank + name); do not append the service number to this field.

                Handwritten capture: include only when the text clearly looks like a service number (prefix letters such as IC/JC/SS etc., digits, optional trailing letter); ignore unrelated scribbles and stamps like 'DUPLICATE'.

                Prefer the value closest to its label when multiple candidates appear; otherwise choose the most prominent printed line in the expected area.

                Preserve original capitalization and spacing for names and numbers; do not expand abbreviations.

                Output must be valid RFC 8259 JSON that parses without errors, with the exact keys and order shown above; use 'Not Found' for missing values."""

        # Run OCR with error handling
        try:
            data = await run_ocr_prompt(prompt, base64_image)
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        # Validate extracted data
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="Invalid OCR response format")

        # Insert with error checking
        try:
            result = ocr_collection.insert_one({
                "user_id": str(current_user["_id"]),
                "doc_type": "echs_card",
                "image_file_id": file_id,
                "extracted_data": data,
                "uploaded_at": datetime.utcnow()
            })
            
            if not result.inserted_id:
                raise HTTPException(status_code=500, detail="Failed to save OCR result")
                
            logger.info(f"OCR result saved with ID: {result.inserted_id}")
                
        except Exception as e:
            logger.error(f"Database insertion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

        return {
            "status": "success", 
            "doc_type": "echs_card", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ECHS card extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/extract/temporary_slip")
async def extract_temporary_slip(
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Temporary slip extraction started for user: {current_user['email']}")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file provided")

        base64_image = base64.b64encode(contents).decode("utf-8")

        # Upload file to GridFS
        try:
            file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        
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
        There is no separate field called "Patient Name".
        If any field is missing, return "Not Found".
        Return only valid JSON.
        """

        try:
            data = await run_ocr_prompt(prompt, base64_image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        # Validate and insert
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail="Invalid OCR response format")

        try:
            result = ocr_collection.insert_one({
                "user_id": str(current_user["_id"]),
                "doc_type": "temporary_slip",
                "image_file_id": file_id,
                "extracted_data": data,
                "uploaded_at": datetime.utcnow()
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

        return {
            "status": "success", 
            "doc_type": "temporary_slip", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 2. Referral Letter Extraction ----------
@app.post("/extract/referral_letter")
async def extract_referral_letter(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Referral letter extraction started for user: {current_user['email']}")
        
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file provided")

        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Upload file to GridFS
        try:
            file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

        prompt = """
        You are analyzing a Referral Letter. Extract:
        - Polyclinic Name
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
        Don't mix up claim ID and Referral Number.
        Referral Number is of 14 digits.
        Return only valid JSON.
        """

        try:
            data = await run_ocr_prompt(prompt, base64_image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        try:
            result = ocr_collection.insert_one({
                "user_id": str(current_user["_id"]),
                "doc_type": "referral_letter",
                "image_file_id": file_id,
                "extracted_data": data,
                "uploaded_at": datetime.utcnow()
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

        return {
            "status": "success", 
            "doc_type": "referral_letter",
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id), 
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 3. Aadhar Card Extraction ----------
@app.post("/extract/aadhar_card")
async def extract_aadhar_card(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Aadhar card extraction started for user: {current_user['email']}")
        
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file provided")

        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Upload file to GridFS
        try:
            file_id = fs.put(contents, filename=file.filename, contentType=file.content_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

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

        try:
            data = await run_ocr_prompt(prompt, base64_image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

        try:
            result = ocr_collection.insert_one({
                "user_id": str(current_user["_id"]),
                "doc_type": "aadhar_card",
                "image_file_id": file_id,
                "extracted_data": data,
                "uploaded_at": datetime.utcnow()
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

        return {
            "status": "success", 
            "doc_type": "aadhar_card", 
            "ocr_result_id": str(result.inserted_id),
            "image_file_id": str(file_id),
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieving the stored image
@app.get("/image/{file_id}")
def get_image(file_id: str, current_user: dict = Depends(get_current_user)):
    try:
        grid_out = fs.get(ObjectId(file_id))
        return StreamingResponse(grid_out, media_type=grid_out.content_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")

# Pydantic models
class FinalSubmissionRequest(BaseModel):
    echs_card_result_id: Optional[str] = None
    referral_letter_result_id: Optional[str] = None
    aadhar_card_result_id: Optional[str] = None
    
class SubmitRequestPayload(BaseModel):
    matched: Optional[bool] = None

# Submit request API (stores OCR doc IDs, not file_ids)
@app.post("/submit_request")
def submit_request(payload: SubmitRequestPayload, current_user: dict = Depends(get_current_user)):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class OCRUpdateRequest(BaseModel):
    extracted_data: dict

@app.get("/ocr/{ocr_result_id}")
def get_ocr_result(
    ocr_result_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        ocr_result = ocr_collection.find_one({"_id": ObjectId(ocr_result_id)})
        if not ocr_result:
            raise HTTPException(status_code=404, detail="OCR result not found")
        if str(ocr_result["user_id"]) != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to view this record")

        ocr_result["_id"] = str(ocr_result["_id"])
        if ocr_result.get("image_file_id"):
            ocr_result["image_file_id"] = str(ocr_result["image_file_id"])
        return {"status": "success", "ocr_result": ocr_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def objid_to_str(doc):
    """Recursively convert all ObjectIds in a dict or list to strings."""
    if isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, dict):
        return {k: objid_to_str(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [objid_to_str(item) for item in doc]
    return doc

# Enhanced update function with comprehensive error handling
@app.put("/request_update/{request_id}")
async def update_request_ocr_results(
    request_id: str,
    payload: dict,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Update request started for request_id: {request_id}")
        
        # Validate request_id format first
        try:
            request_obj = requests_collection.find_one({"_id": ObjectId(request_id)})
        except Exception as e:
            logger.error(f"Invalid request_id format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid request_id format: {str(e)}")

        if not request_obj:
            raise HTTPException(status_code=404, detail="Request not found")

        if str(request_obj["user_id"]) != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to edit this request")

        updates = payload.get("updates", [])
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        # Step 1: Apply updates with comprehensive validation
        update_results = []
        for update in updates:
            doc_type = update.get("doc_type")
            extracted_data = update.get("extracted_data", {})

            if not doc_type or not extracted_data:
                logger.warning(f"Skipping update due to missing doc_type or extracted_data")
                continue

            # Map request_obj field names with doc_type
            ocr_id = None
            if doc_type == "echs_card":
                ocr_id = request_obj.get("echs_card_result_id")
            elif doc_type == "referral_letter":
                ocr_id = request_obj.get("referral_letter_result_id")
            elif doc_type == "aadhar_card":
                ocr_id = request_obj.get("aadhar_card_result_id")
            else:
                logger.warning(f"Unknown doc_type: {doc_type}")
                continue

            if not ocr_id:
                logger.warning(f"No OCR ID found for doc_type: {doc_type}")
                continue

            # Validate ObjectId
            try:
                ocr_object_id = ObjectId(ocr_id)
            except Exception as e:
                logger.error(f"Invalid OCR ID format: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid OCR ID format: {str(e)}")

            # Prepare update fields
            set_fields = {f"extracted_data.{k}": v for k, v in extracted_data.items()}
            set_fields["updated_at"] = datetime.utcnow()

            logger.info(f"Attempting to update OCR ID: {ocr_id} with data: {extracted_data}")

            # Perform update with result checking
            try:
                result = ocr_collection.update_one(
                    {"_id": ocr_object_id},
                    {"$set": set_fields}
                )

                logger.info(f"Update result - matched: {result.matched_count}, modified: {result.modified_count}")

                # Check if update was successful
                if result.matched_count == 0:
                    raise HTTPException(status_code=404, detail=f"OCR document with ID {ocr_id} not found")
                
                if result.modified_count == 0:
                    # Document exists but wasn't modified (values might be the same)
                    logger.warning(f"Document {ocr_id} matched but not modified - values may be identical")

                update_results.append({
                    "ocr_id": ocr_id,
                    "doc_type": doc_type,
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                    "status": "success" if result.modified_count > 0 else "no_changes"
                })

            except Exception as e:
                logger.error(f"Database update failed for {ocr_id}: {str(e)}")
                update_results.append({
                    "ocr_id": ocr_id,
                    "doc_type": doc_type,
                    "status": "error",
                    "error": str(e)
                })

        # Step 2: Fetch ALL docs (updated + not updated) for verification
        all_docs = []
        for doc_type, field_name in {
            "echs_card": "echs_card_result_id",
            "referral_letter": "referral_letter_result_id",
            "aadhar_card": "aadhar_card_result_id"
        }.items():
            ocr_id = request_obj.get(field_name)
            if not ocr_id:
                continue

            try:
                doc = ocr_collection.find_one({"_id": ObjectId(ocr_id)}, {"extracted_data": 1, "updated_at": 1})
                if doc:
                    all_docs.append({
                        "ocr_id": str(ocr_id),
                        "doc_type": doc_type,
                        "extracted_data": doc.get("extracted_data", {}),
                        "updated_at": doc.get("updated_at")
                    })
            except Exception as e:
                logger.error(f"Failed to fetch document {ocr_id}: {str(e)}")

        return {
            "status": "success",
            "request_id": request_id,
            "total_docs": len(all_docs),
            "update_results": update_results,
            "docs": all_docs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_request_ocr_results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@app.post("/generate_claim_id")
def generate_claim_id():
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            # Fetch latest referral
            referral = ocr_collection.find_one(sort=[("_id", -1)])
            if not referral:
                raise Exception("No referral data found in MongoDB!")

            # Go to login page
            page.goto("https://www.echsbpa.utiitsl.com/ECHS/")
            page.fill("#username", "parashos")
            page.fill("#password", "Paras@123")

            # --- CAPTCHA Handling ---
            captcha_selector = "#img_captcha"
            page.wait_for_selector(captcha_selector)
            captcha_buffer = page.locator(captcha_selector).screenshot()
            base64_image = f"data:image/png;base64,{base64.b64encode(captcha_buffer).decode()}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the digits from this CAPTCHA image. Only return the numbers."},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ],
                }]
            )

            captcha_text = response.choices[0].message.content
            captcha_text = re.sub(r"[^0-9]", "", captcha_text).strip()
            page.fill("#txtCaptcha", captcha_text)
            page.get_by_role("button", name="Sign In").click()
            page.wait_for_timeout(3000)

            # Close popup if exists
            try:
                popup_selector = 'button:has-text("Close")'
                if page.locator(popup_selector).is_visible():
                    page.click(popup_selector)
            except:
                pass

            # Fill referral data
            page.locator("#ihaveseennmi").check()
            page.get_by_role("link", name="Intimation").click()
            page.get_by_role("link", name="Accept Referral").click()

            referral_no = referral["extracted_data"]["Referral No"]
            center_code = referral_no[:4]
            trimmed_referral = referral_no[4:]

            center_map = {
                "0147": "5037",
                "0144": "5036",
                "0142": "5033",
                "0143": "5031",
                "0146": "5038",
                "0145": "5042",
                "0431": "5076",
                "0148": "5043"
            }

            if center_code in center_map:
                page.locator("#referredDispensary").select_option(center_map[center_code])
            else:
                raise Exception(f"Center code {center_code} not mapped!")

            page.locator("(//input[@name='cardnum2'])[1]").fill(trimmed_referral)
            page.locator("(//input[@name='serviceNo'])[1]").fill(referral["extracted_data"]["Service No"])

            page.get_by_role("button", name="Search").click()
            page.wait_for_timeout(6000)

            # Select Yes and Submit
            page.wait_for_selector("input[type='radio'][value='Y']")
            page.locator("input[type='radio'][value='Y']").check()
            page.locator("input#submit_save").click()
            page.wait_for_timeout(3000)

            # Extract claim ID from popup
            popup_selector2 = "#ws_alert_dialog"
            page.wait_for_selector(popup_selector2, state="visible")
            popup_text = page.locator(popup_selector2).inner_text()

            match = re.search(r"claim id\s*-\s*(\d+)", popup_text, re.IGNORECASE)
            claim_id = match.group(1) if match else None

            if not claim_id:
                raise Exception("Claim ID not found!")

            # Update in MongoDB
            ocr_collection.update_one(
                {"_id": referral["_id"]},
                {"$set": {"extracted_data.Claim ID": claim_id}}
            )

            browser.close()

        return {"status": "success", "claim_id": claim_id}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e) or repr(e),
            "type": e.__class__.__name__,
            "traceback": traceback.format_exc()
        }

@app.get("/search/{referral_no}")
def search_referral_letter(
    referral_no: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Search in referral letters of the logged-in user
        query = {
            "doc_type": "referral_letter",
            "user_id": str(current_user["_id"]),
            "extracted_data.Referral No": referral_no
        }

        result = ocr_collection.find_one(query)

        if not result:
            raise HTTPException(status_code=404, detail="Referral letter not found")

        # Convert ObjectIds to strings for response
        result["_id"] = str(result["_id"])
        if result.get("image_file_id"):
            result["image_file_id"] = str(result["image_file_id"])

        return {"status": "success", "referral_letter": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get history for logged-in user
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    try:
        history = list(requests_collection.find({"user_id": str(current_user["_id"])}))
        for h in history:
            h["_id"] = str(h["_id"])
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/user_history/{user_id}")
def get_user_history(user_id: str):
    logger.info(f"Admin requesting history for user_id: {user_id}")
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

@app.get("/admin/export_patient_data")
def export_patient_data():
    try:
        # Fetch all OCR docs
        docs = list(ocr_collection.find({}))
        
        # Flatten documents for CSV
        rows = []
        for d in docs:
            rows.append({
                "patient_user_id": str(d.get("user_id", "")),
                "doc_type": d.get("doc_type", ""),
                "uploaded_at": d.get("uploaded_at", ""),
                **d.get("extracted_data", {})  # flatten extracted fields
            })
        
        if not rows:
            raise HTTPException(status_code=404, detail="No patient data found")

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Save as CSV
        file_path = "patient_data_export.csv"
        df.to_csv(file_path, index=False)

        return FileResponse(
            path=file_path,
            filename="patient_data_export.csv",
            media_type="text/csv"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/prod")
def get_prod():
    return {"message": "hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
