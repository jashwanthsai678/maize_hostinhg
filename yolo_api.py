import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="YOLO Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_MODEL_PATH = "Yolo_pest_identification.pt"

try:
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load YOLO model. Error: {e}")
    yolo_model = None

@app.get("/")
def read_root():
    return {"message": "YOLO Inference Service is running", "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": yolo_model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model is not loaded.")
    
    try:
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        results = yolo_model(img, verbose=False)
        result = results[0]
        
        diagnosis_class = "healthy"
        confidence = 0.0
        severity = "low"
        
        # Handle classification VS detection dynamically
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
            
            # severity proxy
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
