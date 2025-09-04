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

load_dotenv()

# MongoDB connection
client_mongo = pymongo.MongoClient(
    os.getenv("MONGO_URL")
)
db = client_mongo[os.getenv("MONGO_NAME")]
users_collection = db["users"]
ocr_collection = db["ocr_results"]
file_collection = db["files"]

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
        print("thi si for register")
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

# OpenAI API key
client.api_key = os.getenv("OPENAI_API")

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
        max_tokens=1500
    )

    reply = response.choices[0].message.content.strip()

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

class OCRUpdateRequest(BaseModel):
    extracted_data: dict

@app.get("/ocr/{ocr_result_id}")
def get_ocr_result(
    ocr_result_id: str,
    current_user: dict = Depends(get_current_user)
):
    ocr_result = ocr_collection.find_one({"_id": ObjectId(ocr_result_id)})
    if not ocr_result:
        raise HTTPException(status_code=404, detail="OCR result not found")
    if str(ocr_result["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to view this record")

    ocr_result["_id"] = str(ocr_result["_id"])
    if ocr_result.get("image_file_id"):
        ocr_result["image_file_id"] = str(ocr_result["image_file_id"])
    return {"status": "success", "ocr_result": ocr_result}

def objid_to_str(doc):
    """Recursively convert all ObjectIds in a dict or list to strings."""
    if isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, dict):
        return {k: objid_to_str(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [objid_to_str(item) for item in doc]
    return doc

@app.put("/request_update/{request_id}")
async def update_request_ocr_results(
    request_id: str,
    payload: dict,
    current_user: dict = Depends(get_current_user)
):
    try:
        request_obj = requests_collection.find_one({"_id": ObjectId(request_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request_id format")

    if not request_obj:
        raise HTTPException(status_code=404, detail="Request not found")

    if str(request_obj["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to edit this request")

    updates = payload.get("updates", [])
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated_docs = []
    for update in updates:
        doc_type = update.get("doc_type")
        extracted_data = update.get("extracted_data", {})

        if not doc_type or not extracted_data:
            continue

        # Map request_obj field names with doc_type
        if doc_type == "echs_card":
            ocr_id = request_obj.get("echs_card_result_id")
        elif doc_type == "referral_letter":
            ocr_id = request_obj.get("referral_letter_result_id")
        elif doc_type == "aadhar_card":
            ocr_id = request_obj.get("aadhar_card_result_id")
        else:
            continue

        if not ocr_id:
            continue

        # Prepare dot-notation updates for partial field update
        set_fields = {f"extracted_data.{k}": v for k, v in extracted_data.items()}
        set_fields["updated_at"] = datetime.utcnow()

        ocr_collection.update_one(
            {"_id": ObjectId(ocr_id)},
            {"$set": set_fields}
        )
        updated_docs.append(str(ocr_id))

    return {
        "status": "success",
        "updated_count": len(updated_docs),
        "request_id": request_id,
        "updated_docs": updated_docs
    }

@app.post("/generate_claim_id")
def generate_claim_id():
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
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

            response = client.responses.create(
                model="gpt-4o-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read the digits from this CAPTCHA image. Only return the numbers."},
                        {"type": "input_image", "image_url": base64_image}
                    ],
                }]
            )

            captcha_text = response.output_text
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
    history = list(requests_collection.find({"user_id": str(current_user["_id"])}))
    for h in history:
        h["_id"] = str(h["_id"])
    return {"status": "success", "history": history}


@app.get("/admin/user_history/{user_id}")
def get_user_history(user_id: str):
    print(user_id)
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