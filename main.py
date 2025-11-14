import os
import json
import base64
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ParseLabelRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image data (no data URI prefix)")
    label_type: Optional[str] = Field(
        default="nutrition",
        description="Type of label: 'nutrition' or 'vitals'"
    )


class ParseLabelResponse(BaseModel):
    name: Optional[str] = None
    serving_size_g: Optional[float] = None
    per_100g: Dict[str, Any] = {}
    per_serving: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if backend is running"""
    return {"backend": "✅ Running"}


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"


SYSTEM_INSTRUCTIONS = (
    "You are a nutrition and supplement label parser. Extract structured data as JSON. "
    "When values are given per serving, try to compute per 100 g if serving mass in grams is present. "
    "Return a compact JSON only, no extra commentary. Keys: name, serving_size_g (number or null), per_100g (object), per_serving (object or null), notes (string). "
    "per_100g should include as available: calories_kcal, protein_g, carbs_g, sugar_g, fiber_g, fat_g, saturated_fat_g, trans_fat_g, sodium_mg, potassium_mg, cholesterol_mg, calcium_mg, iron_mg, vitamin_c_mg, vitamin_d_ug, vitamin_b12_ug, magnesium_mg, zinc_mg, others as found. "
)


def call_gemini(image_b64: str, label_type: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or ""  # Will error if missing
    if not api_key:
        # Fallback to user-provided key if set via environment at runtime
        # Keeping empty here triggers an informative HTTPException
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_INSTRUCTIONS + f" Label type: {label_type}. Output pure JSON."},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_b64,
                        }
                    }
                ]
            }
        ]
    }

    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(GEMINI_ENDPOINT, params=params, headers=headers, data=json.dumps(payload), timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {str(e)}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Gemini error: {resp.text[:500]}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected Gemini response format")

    # Attempt to extract JSON from the model output
    text_stripped = text.strip()
    # Handle code fences
    if text_stripped.startswith("```"):
        text_stripped = text_stripped.strip("`")
        # Remove potential json language hint
        if text_stripped.lower().startswith("json"):
            text_stripped = text_stripped[4:]
    text_stripped = text_stripped.strip()

    try:
        parsed = json.loads(text_stripped)
    except json.JSONDecodeError:
        # Try to locate the first and last braces
        start = text_stripped.find("{")
        end = text_stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text_stripped[start:end+1])
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to parse JSON from Gemini output")
        else:
            raise HTTPException(status_code=500, detail="No JSON found in Gemini output")

    return parsed


@app.post("/api/parse-label", response_model=ParseLabelResponse)
def parse_label(req: ParseLabelRequest):
    parsed = call_gemini(req.image_base64, req.label_type or "nutrition")

    # Normalize keys
    name = parsed.get("name")
    serving_size_g = parsed.get("serving_size_g")
    per_100g = parsed.get("per_100g") or {}
    per_serving = parsed.get("per_serving")

    resp = {
        "name": name,
        "serving_size_g": serving_size_g,
        "per_100g": per_100g,
        "per_serving": per_serving,
        "notes": parsed.get("notes"),
        "raw": parsed,
    }

    return resp


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
