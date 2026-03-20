import asyncio
import json
import os
import httpx
import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# Import custom prediction logic
from blended_predictor import predict_blended_yield

load_dotenv()

app = FastAPI(title="AgriSense AI - Unified Advisory System", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
LLM_API_URL = os.getenv("LLM_API_URL", "http://34.69.210.248:8000/advisory")
YOLO_MODEL_PATH = "Yolo_pest_identification.pt"

# Load YOLO model at startup
try:
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load YOLO model. Error: {e}")
    yolo_model = None

# --- Models ---
class YieldMetadata(BaseModel):
    district_std: str
    crop_year: int
    season: str
    area_ha: float
    T2M: list[float]
    T2M_MAX: list[float]
    T2M_MIN: list[float]
    PRECTOTCORR: list[float]
    RH2M: list[float]
    ALLSKY_SFC_SW_DWN: list[float]
    crop_type: str = "maize"
    growth_stage: str = "vegetative"
    language: str = "english"

# --- Internal Prediction Logic ---

async def run_yolo_diagnosis_internal(image_bytes: bytes) -> dict:
    """Internal YOLO logic without HTTP call."""
    if yolo_model is None:
        return {"diagnosis": "YOLO model not loaded", "confidence": 0, "severity": "unknown", "annotated_image_base64": ""}
    
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = yolo_model(img, verbose=False)
        result = results[0]
        
        diagnosis_class = "healthy"
        confidence = 0.0
        severity = "low"
        
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            top_class_idx = probs.argmax()
            confidence = float(probs[top_class_idx])
            diagnosis_class = result.names[top_class_idx]
            if confidence > 0.8: severity = "high"
            elif confidence > 0.5: severity = "medium"
            else: severity = "low"
        elif len(result.boxes) > 0:
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            diagnosis_class = result.names[cls_id]
            confidence = float(best_box.conf[0])
            
            total_box_area = sum(float(b.xywh[0][2] * b.xywh[0][3]) for b in result.boxes)
            img_area = img.width * img.height
            area_ratio = total_box_area / img_area
            if area_ratio > 0.3: severity = "high"
            elif area_ratio > 0.1: severity = "medium"
            else: severity = "low"
        else:
            diagnosis_class = "no issues detected"
            
        try:
            annotated_img_numpy = result.plot()
            # Convert BGR to RGB
            annotated_img = Image.fromarray(annotated_img_numpy[..., ::-1])
        except:
            annotated_img = img
            
        buffered = BytesIO()
        annotated_img.save(buffered, format="JPEG")
        annotated_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "diagnosis": diagnosis_class.lower().replace("_", " "),
            "confidence": round(confidence, 4),
            "severity": severity,
            "annotated_image_base64": annotated_b64
        }
    except Exception as e:
        print(f"Internal YOLO Error: {e}")
        return {"diagnosis": "Error running YOLO", "confidence": 0, "severity": "unknown", "annotated_image_base64": ""}

async def get_llm_advisory(payload: dict) -> dict:
    """Call the LLM API on the cloud VM."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"LLM Error: {e}")
        return {
            "advisory": "### 💡 Expert Advisory (Fallback Mode)\n\n"
                        "We are currently experiencing a connection issue with the remote AI Advisor. "
                        "However, based on the processed diagnostic data:\n\n"
                        "- **Immediate Action**: Scrutinize the affected area and remove heavily infested leaves.\n"
                        "- **Preventive**: Maintain soil health and monitor neighboring plots.",
            "error": str(e)
        }

# --- Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('index.html')

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.post("/orchestrate")
async def orchestrate_advisory(
    image: UploadFile = File(...),
    metadata: str = Form(...)
):
    try:
        # 1. Parse Metadata
        try:
            meta_dict = json.loads(metadata)
            meta = YieldMetadata(**meta_dict)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata format: {e.errors()}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata must be valid JSON")

        image_bytes = await image.read()

        # 2. Run Yield Model (Direct function call)
        try:
            yield_result = predict_blended_yield(meta.model_dump())
        except Exception as e:
            print(f"Yield Prediction Error: {e}")
            yield_result = {"pred_best_rmse_blend": "Error computing yield"}

        # 3. Run YOLO (Internal function)
        yolo_result = await run_yolo_diagnosis_internal(image_bytes)

        # 4. Prepare payload for LLM Interpretation
        expected_yield_val = yield_result.get("pred_best_rmse_blend", "Unknown")
        expected_yield_str = f"{expected_yield_val:.2f} t/ha" if isinstance(expected_yield_val, (float, int)) else expected_yield_val

        llm_input = {
            "crop": meta.crop_type,
            "growth_stage": meta.growth_stage,
            "district": meta.district_std,
            "season": meta.season,
            "diagnosis": yolo_result.get("diagnosis", "unknown"),
            "confidence": yolo_result.get("confidence", 0),
            "severity": yolo_result.get("severity", "unknown"),
            "expected_yield": expected_yield_str,
            "language": meta.language
        }
        
        # 5. Run LLM call
        llm_result = await get_llm_advisory(llm_input)
        
        # 6. Final Aggregation
        return {
            "status": "success",
            "visual_diagnosis": yolo_result,
            "environmental_context": {
                "district": meta.district_std,
                "season": meta.season,
                "expected_yield_baseline": expected_yield_str,
                "is_error": "Error" in str(expected_yield_val)
            },
            "expert_advisory": llm_result
        }
        
    except Exception as e:
        print(f"Orchestration Global Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
