import asyncio
import sys
import base64, re, traceback
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())



from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
import ssl



from typing import List
from PIL import Image
import io, base64
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
from fastapi import FastAPI
from playwright.sync_api import sync_playwright

app = FastAPI()
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



user_accounts = {
    "user1": {
        "username": "parashos",
        "password": "Paras@12",
        "polyclinics": ["0147", "0144", "0142", "0143", "0146", "0145", "0431", "0148"]
    },
    "user2": {
        "username": "parasggn",
        "password": "Paras@12",
        "polyclinics": ["0149","0150", "0151", "0152","0153", "0154"]
    }
}

# --- Center mapping for dropdown values ---
center_map = {
    "0147": "5037",
    "0144": "5036",
    "0142": "5033",
    "0143": "5031",
    "0146": "5038",
    "0145": "5042",
    "0431": "5076",
    "0148": "5043",
    "0150": "5293",
    "0151": "5294",
    "0152": "5290",
    "0149": "5292",
    "0153": "5329",
    "0154": "5296"
}


def get_account_for_poly(poly_id: str):
    """Return the account details for the given polyclinic"""
    for acc, details in user_accounts.items():
        if poly_id in details["polyclinics"]:
            return details
    return None






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
        prompt = """You are extracting structured data from an ECHS smart card image. Read all printed and handwritten text and return one JSON object that strictly follows the schema and rules below. No extra keys, comments, or prose. If a value is missing or illegible, output exactly 'Not Found'. Return only JSON.

Schema (key order must be preserved):
{
  "Card No": string, // Printed card/reg number on the card face (e.g., 'JB 0000 0268 6390', 'HI 0000 0242 8145'); preserve spaces exactly as shown.
  "Patient Name": string, // Name at the top-left header; transcribe exactly as printed.
  "ESM": string, // ESM field is the printed name of the ex-serviceman only; include rank and name exactly as shown (do not include service number here). Look for a specific ESM field on the card. If no separate ESM field exists and the card holder is the ESM themselves, use 'Not Found'.
  "Relationship with ESM": string, // Relationship term on the card such as 'Spouse', 'Son', 'Daughter', 'Self', etc. If the card holder is the ESM themselves, use 'Self'. If absent or unclear, use 'Not Found'.
  "DOB": string, // Value labeled 'DOB' (Date of Birth); normalize to DD MMM YYYY format when month is textual (e.g., '01 Jun 1956'); otherwise keep as printed.
  "DOM": string, // Value labeled 'DOM' (Date of Membership); same normalization rule as DOB; if absent, use 'Not Found'.
  "Service No": string // Service number of the ESM: prefer a printed value labeled 'Service No'; if not present, use clear handwritten alphanumeric text near the name/photo area that matches common military service number patterns (e.g., 'JC257424Y', 'IC 12345', 'SS-12345'); preserve case, spaces, and hyphens exactly. If both printed and handwritten exist, use the printed one. If none found, use 'Not Found'.
}

               Extraction rules:

1. ESM Field Logic: Look for a specific 'ESM' field on the card. If the card holder is the ex-serviceman themselves (indicated by military rank in Patient Name), the ESM field may not be separately printed. In such cases, use 'Not Found' for ESM field.

2. Relationship Logic: If the Patient Name contains a military rank (like 'SUB', 'LT', 'COL', etc.), the relationship is typically 'Self'. Otherwise, look for explicit relationship terms.

3. Service Number Identification: Military service numbers typically follow patterns like:
   - Army: JC######Y, IC#####, SS#####
   - Navy: Starts with letters followed by numbers
   - Air Force: Numbers with letter suffix
   Include only clear, complete service numbers that match these patterns.

4. Text Preservation: Maintain original capitalization, spacing, and punctuation exactly as shown on the card.

5. Date Formatting: Convert dates to DD MMM YYYY format only when the month appears as text (Jan, Feb, etc.). Keep numeric dates as printed.

6. Missing Data: Use 'Not Found' for any field that is not visible, illegible, or clearly absent from the card.

Output must be valid RFC 8259 JSON that parses without errors, with the exact keys and order shown above."""
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
    file: UploadFile = File(None), #when testing postman file name should same (when get err 500 )
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
                You are analyzing an ECHS temporary receipt document. Extract ONLY these 8 fields exactly as seen:

EXTRACTION RULES:
1. Form No: Extract pattern like "730138F" - 6-7 digits followed by single letter (F, H, B, etc.). Located in "Received documents from No [FORM_NO]" section.

2. Registration No: Extract 10-digit number like "0002065787" from "Registration No :" field.

3. Patient Name: Extract dependent's name from photo section (Son/Daughter/Spouse name).

4. ESM: Extract Ex-Serviceman's name from "Rank [RANK] Name [ESM_NAME]" section.

5. DOB: Extract dependent's date of birth from photo section in DD Mon YYYY format.

6. Relationship with ESM: Extract relationship (Son/Daughter/Spouse) from photo section.

7. Valid Upto: Extract validity date (may be handwritten).

8. Category of Ward: Extract ward category (Semi-Private/General/Other).

SPECIFIC PATTERNS:
- Form No: [digits][letter] like "730138F", "145644H"  
- Registration No: 10-digit number like "0002065787"
- Dates: DD Mon YYYY format (01 Jun 1967, 06 Jan 1997)

OUTPUT FORMAT:
{
    "Form No": "",
    "Registration No": "", 
    "Patient Name": "",
    "ESM": "",
    "DOB": "",
    "Relationship With Esm": "",
    "Valid Upto": "",
    "Category of Ward": ""
}

If any field is not found or unclear, use "Not Found" as the value.
Return only valid JSON with these exact 8 fields.


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
You are analyzing a Referral Letter. CRITICAL: You have been making consistent errors with specific fields and characters. Follow these rules exactly.

KNOWN ERROR PATTERNS TO AVOID:
1. REFERRAL NO ERROR: You frequently read "01440000402039" as "01440000402319" 
   - This happens because you insert an extra "1" and change "0" to "3"
   - The pattern "...402039" should NEVER become "...402319"
   - If you see 15+ digits, you've made a segmentation error

2. SERVICE NO ERROR: You sometimes confuse Service No with other numbers
   - Service No format: 8-10 digits + 1 letter (e.g., "477762440Y")
   - Do NOT use Contact Numbers, Card Numbers, or Referral Numbers

3. AGE/DECIMAL POINT ERROR: You read decimal points as letters
   - "52.8" becomes "52 B" - WRONG
   - "45.5" becomes "45 S" - WRONG  
   - "38.2" becomes "38 Z" - WRONG
   - Pattern: [number].[digit] should NEVER become [number] [letter]

CHARACTER RECOGNITION RULES:
Pay special attention to these character confusions:
- Decimal point (.) vs letter B, S, 8, O
- Number 0 (zero) vs letter O
- Number 1 vs letter I or l
- Number 5 vs letter S
- Number 6 vs letter G or 9
- Number 8 vs letter B
- Number 2 vs letter Z

DECIMAL NUMBER DETECTION:
- If you see [digit][space][letter] pattern in numeric fields, it's likely a decimal point error
- Example: "52 B" should be "52.8", "45 S" should be "45.5"
- Context check: Age, measurements, scores are often decimals, not "number + letter"

CRITICAL EXTRACTION RULES:
- Referral No: Must be EXACTLY 14 digits from field labeled 'Referral No'
- Service No: Must be digits+letter from field labeled 'Service No' 
- Age: Should be numeric (can include decimals), not "number + letter"
- Read each character individually to prevent segmentation/recognition errors
- For numeric fields, verify decimal points aren't read as letters

FIELD-SPECIFIC INSTRUCTIONS:
Referral No: After extraction, double-check positions 11-14. If you see "2319" at end, it's likely wrong - should be "2039"
Service No: Look for explicit 'Service No' label, usually ends with Y, X, A, or similar letter
Age: Should be purely numeric (e.g., "52.8", "45", "62.5") - if you see "52 B", it should be "52.8"
Clinical Notes: Extract ONLY from 'Clinical Notes' section - never mix with Admission data
Admission: Extract ONLY from 'Admission' section - never use Clinical Notes data
Investigation: Extract ONLY from 'Investigation' section
Referred To: Extract ONLY from 'Referred To' section

Extract the fields below exactly as named and return one JSON object that follows the schema and rules strictly. Do not include extra keys, comments, or prose. If any field is missing or illegible, return exactly 'Not Found'. Return only valid JSON.

Schema (key order fixed):
{
"Polyclinic Name": string, // Header showing the Polyclinic name/location; copy as printed.
"Name of Patient": string, // Field 'Name of Patient'; copy verbatim.
"Referral No": string, // CRITICAL: 14-digit number only. Check for "402319" error pattern.
"Valid Upto": string, // Value labeled 'Validity Upto' or 'Valid Upto'; keep format as printed.
"Date of Issue": string, // Value labeled 'Date Of Issue'; keep as printed.
"No of Sessions Allowed": string, // Field 'No. Of Session Allowed'; copy as printed.
"Patient Type": string, // OPD/IPD etc.; copy as printed.
"Age": string, // CRITICAL: Numeric only (can have decimals). If you see "52 B" it should be "52.8".
"Gender": string, // Gender value; copy as printed.
"Relationship with ESM": string, // Relationship with ESM; copy as printed.
"Category": string, // Category; copy as printed.
"Service No": string, // CRITICAL: From 'Service No' field only. Format: digits+letter.
"Card No": string, // 'Card No' value; preserve spacing.
"ESM Name": string, // 'ESM Name' on the form; copy verbatim.
"ESM Contact Number": string, // 'ESM Contact Number'; copy digits/spaces exactly.
"Clinical Notes": string, // 'Clinical Notes' section ONLY; replace internal newlines with single space.
"Referred To": string, // 'Referred To' section ONLY; copy verbatim.
"Admission": string, // 'Admission' section ONLY; never use Clinical Notes data here.
"Investigation": string, // 'Investigation' section ONLY; copy verbatim.
"Consultation For": string, // 'Consultation For' field value; copy verbatim.
"Polyclinic Remarks": string, // 'Polyclinic Remarks' field; copy verbatim.
"Claim ID": string // Value labeled 'Claim ID' only; never use Referral No.
}

VALIDATION STEPS:
1. After extracting Referral No, count digits (must be exactly 14)
2. Check for known error pattern "402319" - should be "402039"
3. Verify Service No has digits+letter format from correct field
4. Check Age field - if format is "number letter" (e.g., "52 B"), convert to decimal (e.g., "52.8")
5. Ensure no data mixing between Clinical Notes and Admission fields
6. For any numeric fields showing "number + single letter", check if it should be decimal

DECIMAL POINT CORRECTIONS:
- "52 B" → "52.8"
- "45 S" → "45.5" 
- "38 Z" → "38.2"
- "67 G" → "67.6"
- Pattern: If numeric field shows [digit][space][letter], likely decimal point error

FINAL CHECK:
- Referral No: Exactly 14 digits, check end digits for "2039" vs "2319" error
- Service No: From Service No field only, ends with letter
- Age: Numeric format only (no letters unless it's actually part of age like "52 years")
- Field Separation: No mixing of Clinical Notes with Admission data
- Character Count: If numbers have wrong digit count, mark as 'Not Found'
- Decimal Recognition: Convert "number letter" patterns to "number.digit" in numeric fields

Rules:
1. FIELD ISOLATION: Extract each field ONLY from its specifically labeled section
2. ERROR PATTERN PREVENTION: Watch for known "402319" vs "402039" confusion
3. DECIMAL POINT ACCURACY: Recognize when dots are misread as letters in numeric fields
4. CHARACTER PRECISION: Read digits individually to prevent segmentation errors
5. DATA INTEGRITY: Never mix content between different labeled sections
6. NUMERIC VALIDATION: Ensure numeric fields contain proper numbers, not "number + letter"
7. Output must be RFC 8259 compliant JSON with exact key names and order above; no comments or trailing commas.

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




@app.post("/extract/prescription")

async def extract_prescription(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validate
        if not files or len(files) > 2:
            raise HTTPException(status_code=400, detail="Upload 1–2 images")

        # Save originals to GridFS and collect bytes
        contents, file_ids = [], []
        for f in files:
            b = await f.read()
            contents.append(b)
            file_ids.append(str(fs.put(b, filename=f.filename, contentType=f.content_type)))

        # Merge-if-two, else use single
        if len(contents) == 1:
            final_bytes = contents[0]
        else:
            from PIL import Image
            import io
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in contents]
            w = max(i.width for i in imgs)
            imgs = [i if i.width == w else i.resize((w, int(i.height * w / i.width))) for i in imgs]
            h = sum(i.height for i in imgs)
            can = Image.new("RGB", (w, h), "white")
            y = 0
            for i in imgs:
                can.paste(i, (0, y)); y += i.height
            buf = io.BytesIO()
            can.save(buf, "JPEG", quality=90)
            final_bytes = buf.getvalue()

        # Final base64 for OCR
        base64_image = base64.b64encode(final_bytes).decode("utf-8")



        # 🔹 Prompt for Prescription with Admission Advice
        prompt = """
Task: Analyze the attached medical prescription(s) and extract structured data as JSON.


Inputs:


One or two images may be provided (Prescription Page 1 and optionally Page 2). Treat them as parts of the same encounter. If only one image exists, proceed with that page. If two images exist, merge information across both pages and resolve duplicates consistently (prefer the more explicit entry; if still uncertain, mark as ‘Unclear from prescription’).


Output JSON schema (return exactly these keys):
{
"name": "Patient Full Name or 'Not specified'",
"age": "Age (with units if written) or 'Not specified'",
"diagnosis": "Complete diagnosis with staging, biomarkers, and dates exactly as written, or 'Not specified'",
"advice": "Doctor’s advice/instructions verbatim (summarize multi-line text into one string) or 'Not specified'",
"medication": [
{ "name": "Drug or brand (include generic if both appear)", "dosage": "Exact strength + route + frequency + duration, e.g., 'paclitaxel 100 mg IV D1' or 'syp Apivite 10 ml BD 30 days'" }
],
"treatment_plan": [
"Step 1 or plan item (e.g., 'CT-RT weekly concurrent chemo')",
"Monitoring/assessments (e.g., 'Order PDL-1 on tumour tissue; NGS 12-gene panel')",
"Follow-up instruction (e.g., 'Re-admit to day care after cardiology review')"
],
"gender": "Gender as written or 'Not specified'"
}


Extraction rules:


Preserve original medical terminology, capitalization, drug names, and abbreviations (CT-RT, D1, BD, IV, etc.).


Combine information from both pages; do not duplicate identical items.


For diagnosis, include stage, histology, site, laterality, biomarkers (e.g., PDL-1), and date on the prescription if present.


For medication, include each distinct item (injections, tablets, syrups) with exact dose, route, schedule, and duration when specified.


For treatment_plan, list therapy steps, investigations, monitoring, referrals (e.g., cardiology review), and sequencing (e.g., ‘then give …’).


If a required value is missing or unreadable, return 'Not specified' or 'Unclear from prescription' (do not infer).


Strip personally identifying numbers except those explicitly part of medical orders (e.g., “PDL-1”, “12 gene panel”).


Return only the JSON object, no extra commentary or Markdown.


Edge cases:


Handwritten ambiguity: prefer exact tokens; if multiple interpretations exist, choose the clearest and note others in parentheses or mark as ‘Unclear from prescription’.


Units: keep units as written (mg, ml, %, tabs, IV, SC, OD, BD, TID, HS, weekly, D1, D8, etc.).


Dates: include the written date(s) in ISO format if clear, else keep as seen (e.g., ‘29/9/25’).


Now read the provided Prescription Page 1 and, if present, Page 2, apply the rules above, and output the single JSON object.
i want to extract gender also / add one more text field is gender that extract gender refer attacted prescription so give me optimized prompt that able to extract gender also - iam giving my current promt not change other but add for gender. i want to extract majorly is what is doctor advised to patient, or what treatment advised / admission or any  that should extract proprly
"""

        # OCR + LLM extraction
        data = await run_ocr_prompt(prompt, base64_image)

        # Save result to MongoDB
        result = ocr_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "doc_type": "prescription",
            "image_file_id": file_ids,
            "extracted_data": data,
            "uploaded_at": datetime.utcnow()
        })

        return {
            "status": "success",
            "doc_type": "prescription",
            "ocr_result_id": str(result.inserted_id),
            # "image_file_id": str(file_id),
            "image_file_id": file_ids if len(file_ids) > 1 else file_ids[0],
            "data": data
        }
 
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
        max_tokens=3000
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
    prescription_result_id: Optional[str] = None
    
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

    prescription = ocr_collection.find_one(
        {"user_id": str(current_user["_id"]), "doc_type": "prescription"},
        sort=[("_id", -1)]
    )

    if not (echs_or_slip or referral or aadhar or prescription):
        raise HTTPException(status_code=400, detail="No documents found for submission")

    request_doc = {
        "user_id": str(current_user["_id"]),
        "echs_card_result_id": str(echs_or_slip["_id"]) if echs_or_slip else None,
        "referral_letter_result_id": str(referral["_id"]) if referral else None,
        "aadhar_card_result_id": str(aadhar["_id"]) if aadhar else None,
        "prescription_result_id": str(prescription["_id"]) if prescription else None,
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

    # Step 1: Apply updates
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
        elif doc_type == "prescription":
            ocr_id = request_obj.get("prescription_result_id")
        else:
            continue

        if not ocr_id:
            continue

        set_fields = {f"extracted_data.{k}": v for k, v in extracted_data.items()}
        set_fields["updated_at"] = datetime.utcnow()

        ocr_collection.update_one(
            {"_id": ObjectId(ocr_id)},
            {"$set": set_fields}
        )

    # Step 2: Fetch ALL docs (updated + not updated)
    all_docs = []

    mapping = {
        "echs_card": "echs_card_result_id",
        "referral_letter": "referral_letter_result_id",
        "aadhar_card": "aadhar_card_result_id",
        "prescription": "prescription_result_id"
    }

    for doc_type, field_name in mapping.items():
        ocr_id = request_obj.get(field_name)
        if not ocr_id:
            continue

        doc = ocr_collection.find_one({"_id": ObjectId(ocr_id)}, {"extracted_data": 1})
        if doc:
            all_docs.append({
                "ocr_id": str(ocr_id),
                "doc_type": doc_type,
                "extracted_data": doc.get("extracted_data", {})
            })

    return {
        "status": "success",
        "request_id": request_id,
        "total_docs": len(all_docs),
        "docs": all_docs
    }








@app.post("/generate_claim_id")
def generate_claim_id():
    try:
        # --- Fetch latest referral from DB ---
        # referral = ocr_collection.find_one(sort=[("_id", -1)])

        referral = ocr_collection.find_one(
            {"extracted_data.Referral No": {"$exists": True}}, 
            sort=[("_id", -1)]
            
        )
        


        if not referral:
            raise Exception("No referral data found in MongoDB!")



        referral_no = referral["extracted_data"]["Referral No"]

        # extracted_data = referral.get("extracted_data", {})


        center_code = referral_no[:4]
        trimmed_referral = referral_no[4:]



        account = get_account_for_poly(center_code)
        if not account:
            raise Exception(f"No account found for polyclinic {center_code}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # --- Login page ---
            page.goto("https://www.echsbpa.utiitsl.com/ECHS/")
            page.fill("#username", account["username"])
            page.fill("#password", account["password"])

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

            # --- Close popup if exists ---
            try:
                popup_selector = 'button:has-text("Close")'
                if page.locator(popup_selector).is_visible():
                    page.click(popup_selector)
            except:
                pass

            # --- Referral Flow ---
            page.locator("#ihaveseennmi").check()
            page.get_by_role("link", name="Intimation").click()
            page.get_by_role("link", name="Accept Referral").click()

            if center_code not in center_map:
                raise Exception(f"Center code {center_code} not mapped!")

            page.locator("#referredDispensary").select_option(center_map[center_code])
            page.locator("(//input[@name='cardnum2'])[1]").fill(trimmed_referral)
            page.locator("(//input[@name='serviceNo'])[1]").fill(referral["extracted_data"]["Service No"])
            page.get_by_role("button", name="Search").click()
            page.wait_for_timeout(6000)

            # --- Check for failure popup ---
            try:
                page.locator("#ws_alert_dialog").wait_for(state="visible", timeout=6000)
                failure_text = page.locator("#ws_alert_dialog #alertpara").inner_text().strip()

                # Save failure message in DB
                ocr_collection.update_one(
                    {"_id": referral["_id"]},
                    {"$set": {"extracted_data.ErrorMessage": failure_text}}
                )
                browser.close()
                return {"status": "error", "message": failure_text}
            except:
                pass

            # --- Success Flow ---
            page.wait_for_selector("input[type='radio'][value='Y']")
            page.locator("input[type='radio'][value='Y']").check()
            page.get_by_text('Submit').click()
            page.wait_for_timeout(3000)

            # --- Extract Claim ID ---
            popup_selector2 = "#ws_success_dialog"
            page.wait_for_selector(popup_selector2, state="visible")
            popup_text = page.locator(popup_selector2).inner_text()

            match = re.search(r"claim id\s*-\s*(\d+)", popup_text, re.IGNORECASE)
            claim_id = match.group(1) if match else None
            if not claim_id:
                raise Exception("Claim ID not found!")

            # Save Claim ID in DB
            ocr_collection.update_one(
                {"_id": referral["_id"]},
                {"$set": {"extracted_data.Claim ID": claim_id}}
            )

            browser.close()
            return {"status": "success", "claim_id": claim_id}

    except Exception as e:
        error_msg = str(e) or repr(e)
        tb = traceback.format_exc()
        # Save error in DB
        ocr_collection.update_one(
            {"_id": referral["_id"]},
            {"$set": {"extracted_data.ErrorMessage": error_msg, "Traceback": tb}}
        )
        return {"status": "error", "message": error_msg, "traceback": tb}









    #Repeate/Follow up

@app.post("/generate_claim_id_followup")
def generate_claim_id_followup():
    try:
        # --- Fetch  referral from DB ---
        # referral = ocr_collection.find_one(sort=[("_id", -1)])
        referral = ocr_collection.find_one(
            {"extracted_data.Referral No": {"$exists": True}}, 
            sort=[("_id", -1)]
            
        )

        if not referral:
            raise Exception("No referral data found in MongoDB!")

        referral_no = referral["extracted_data"]["Referral No"]
        center_code = referral_no[:4]

        account = get_account_for_poly(center_code)
        if not account:
            raise Exception(f"No account found for polyclinic {center_code}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            # --- Login page ---
            page.goto("https://www.echsbpa.utiitsl.com/ECHS/")
            page.fill("#username", account["username"])
            page.fill("#password", account["password"])

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

            # --- Close popup if exists ---
            try:
                popup_selector = 'button:has-text("Close")'
                if page.locator(popup_selector).is_visible():
                    page.click(popup_selector)
            except:
                pass

            # --- Followup Flow ---
            page.locator("#ihaveseennmi").check()
            page.get_by_role("link", name="Intimation").click()
            page.get_by_role("link", name="Followup Referral").click()

            # Referral number full (no trimming)
            page.locator("#referenceNumber").fill(referral_no)
            
            page.get_by_role("button", name="Submit").click()
            page.wait_for_timeout(6000)



            # --- Extract Old Claim ID popup ---
            old_claim_popup = "#ws_info_dialog"
            if page.locator(old_claim_popup).is_visible():
                popup_text = page.locator(f"{old_claim_popup} #infopara").inner_text().strip()
                match = re.search(r"old claim id\s*-\s*(\d+)", popup_text, re.IGNORECASE)
                old_claim_id = match.group(1) if match else None

                if old_claim_id:
                    ocr_collection.update_one(
                        {"_id": referral["_id"]},
                        {"$set": {"extracted_data.Old Claim ID": old_claim_id}}
                    )
            page.wait_for_timeout(6000)
                # Close old claim popup

            page.locator("//button[@class='ui-button ui-corner-all ui-widget']").click()    
            # page.click(f"{old_claim_popup} button:has-text('Close')")
            page.wait_for_timeout(6000)
            # --- Select Yes and In-patient ---
            page.select_option("#confirmAdmit", "Y")
            page.select_option("select[name='revisitPatientType']", "I")  # In-patient
            page.wait_for_timeout(3000)



            # Handle confirmation dialog (OK button)
            page.once("dialog", lambda dialog: dialog.accept())

            # Click the submit button (triggers popup)
            # page.click("button#submit")   # <-- replace with your locator
            try:
        # First attempt: standard button locator
               page.get_by_role("button", name="Submit").click(timeout=5000)
               print("Clicked using first locator")
            except TimeoutError:
                try:
            # Second attempt: using section or text locator as fallback
                    section = page.locator("section#form-section")  # example section
                    section.locator("text=Submit").click(timeout=5000)
                    print("Clicked using section locator fallback")
                except TimeoutError:
                    print("Failed to click the Submit button using both locators")

            # page.locator("//input[@onclick='return validateRepeat();'']").click()
            # page.get_by_role("button", name="Submit").click()


            # page.wait_for_timeout(3000)
            # Accept dialog if appears
            # try:
            #     page.on("dialog", lambda dialog: dialog.accept())
            # except:
            #     pass

            page.wait_for_timeout(3000)


            # --- Extract New Claim ID popup ---
            popup_selector2 = "#ws_Success_dialog"
            page.wait_for_selector(popup_selector2, state="visible")
            popup_text = page.locator(popup_selector2).inner_text()

            match = re.search(r"claim id\s*-\s*(\d+)", popup_text, re.IGNORECASE)
            claim_id = match.group(1) if match else None
            if not claim_id:
                raise Exception("New Claim ID not found!")

            # Save new Claim ID in DB    # update here code for new claim id 
            ocr_collection.update_one(
                {"_id": referral["_id"]},
                {"$set": {"extracted_data.Claim ID": claim_id}}
            )

            browser.close()
            return {
                "status": "success",
                "old_claim_id": old_claim_id if 'old_claim_id' in locals() else None,
                "new_claim_id": claim_id
            }

    except Exception as e:
        error_msg = str(e) or repr(e)
        tb = traceback.format_exc()
        try:
            ocr_collection.update_one(
                {"_id": referral["_id"]},
                {"$set": {"extracted_data.ErrorMessage": error_msg, "Traceback": tb}}
            )
        except:
            pass
        return {"status": "error", "message": error_msg, "traceback": tb}


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
            attach_ocr_data("prescription_result_id", "prescription")

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
    



@app.get("/echs_data")
def echs_data(limit:Optional[int]=None):
        # MongoDB connection string
    CONNECTION_STRING = "mongodb+srv://pilot:pilot@cluster1.rkupr.mongodb.net/?retryWrites=true&w=majority"

    # Connect to MongoDB
    client = MongoClient(CONNECTION_STRING)

    try:
        # Print available databases and collections
        print("Databases:", client.list_database_names())
        db = client.hospital_app
        print("Collections in hospital_app:", db.list_collection_names())

        # Load collections
        df = pd.DataFrame(list(db["ocr_results"].find()))
        df2 = pd.DataFrame(list(db["fs.files"].find()))
        df1 = pd.DataFrame(list(db["requests_history"].find()))

    finally:
        client.close()
        print("MongoDB connection closed.")
        print("", df.shape)
        print("", df2.shape)
        print("", df1.shape)
    df1['echs_card_result_id'] = df1['echs_card_result_id'].astype(str)
    df1['referral_letter_result_id'] = df1['referral_letter_result_id'].astype(str)
    df1['prescription_result_id'] = df1['prescription_result_id'].astype(str)
    df['_id'] = df['_id'].astype(str)
    main_df = df1[["echs_card_result_id","referral_letter_result_id","prescription_result_id"]].drop_duplicates()
    last_df = pd.merge(main_df, df, left_on='echs_card_result_id',right_on='_id', how='inner')
    last_df = last_df.rename(columns={'extracted_data': 'echs_data','doc_type':'echs_card'})
    last_df = pd.merge(last_df, df, left_on='referral_letter_result_id',right_on='_id', how='inner')
    last_df = last_df.rename(columns={'extracted_data': 'referral_letter_data','doc_type':'referral_letter','uploaded_at_x':'echs_upload_date','uploaded_at_y':'refferal_upload_date','image_file_id_x':'echs_image_id','image_file_id_y':'refferal_image_id',})
    final = last_df[['echs_image_id','echs_upload_date','echs_data','refferal_image_id'	,'referral_letter_data'	,'refferal_upload_date']]
    df3 = df2[['filename','_id']]
    final = pd.merge(final, df3, left_on='echs_image_id',right_on='_id', how='inner')
    final = pd.merge(final, df3, left_on='refferal_image_id',right_on='_id', how='inner')
    final = final.rename(columns = {'filename_x' : 'echs_img','filename_y' : 'refferal_img'})
    final = final[['echs_upload_date','echs_data','referral_letter_data'	,'refferal_upload_date','echs_img','refferal_img']]

    result = final.to_dict(orient='records')
    return {"echs_data": result}






@app.get("/prod")
def get_prod():
    return {"message": "hello"}
#