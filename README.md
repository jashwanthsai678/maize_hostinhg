# Orissa Maize Yield Prediction & Pest Advisory System

This project provides an end-to-end agricultural advisory system for maize crops in Orissa, combining:
- **YOLO-based pest/disease detection** from uploaded images
- **Machine learning yield prediction** based on district, season, and weather data
- **LLM-powered advisory** for actionable recommendations

## 🏗️ System Architecture & Flow

The system consists of **two separate inference services behind one advisory orchestrator**.

### How It Functions

1. **User Uploads a Crop Image**
   - YOLO model runs first and returns:
     - Disease/pest/deficiency class
     - Confidence score
     - Affected area/severity proxy
     - Annotated image (with detections highlighted)
   - **This answers: "What is wrong right now?"**

2. **Advisory Layer Interprets YOLO Output**
   - A rule layer or LLM takes:
     - Crop type (e.g., maize)
     - Growth stage (e.g., vegetative)
     - District (e.g., Anugul)
     - Season (e.g., Autumn)
     - YOLO diagnosis, confidence, severity
     - User language (e.g., English)
   - Generates:
     - Likely issue explanation
     - Immediate corrective actions
     - Preventive actions
     - Urgency level
     - When to escalate to agronomist
     - Voice/text output in Indian language
   - **This answers: "What should I do now?"**

3. **Benchmark Yield Model Runs in Parallel**
   - Takes: district_std, crop_year, season
   - Predicts expected yield band/expected yield
   - **This answers: "What is the likely productivity context for this crop in this location/season?"**

4. **Final Advisory Combines Both**
   - Produces output like:
     ```
     Detected issue: leaf blight
     Confidence: high
     Likely severity: medium
     Expected district-season yield baseline: 1.8 t/ha
     Likely risk: yield may fall below normal if untreated
     Recommended next actions: fungicide/nutrient/irrigation/scouting action
     Prevention advice
     Monitor in 5 days
     Contact agronomist if spread crosses threshold
     ```

### Benefits to Farmers
- **Immediate**: Fast diagnosis from image, no expert visit needed.
- **Practical**: Actionable advice on what to do, inputs needed, urgency, prevention.
- **Strategic**: Expected yield context for better prioritization (e.g., "This issue may push yield below normal if untreated").

Instead of just "this is disease X", it provides context: "this is disease X, severity moderate, untreated spread may reduce expected performance."

## 📋 Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- Internet access (for LLM API calls)

## 🚀 Installation

### 1. Clone/Download the Repository
Place the project files in a folder, e.g., `orissa_maize_v2_Yield_Prediction_bundle`.

### 2. Create Virtual Environment
```bash
cd orissa_maize_v2_Yield_Prediction_bundle
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn requests ultralytics pillow pandas numpy scikit-learn joblib python-multipart
```

**Note**: If scikit-learn version conflicts occur, install the specific version:
```bash
pip install "scikit-learn==1.6.1" --upgrade --force-reinstall
```

### 4. Configure LLM API (Optional)
In `orchestrator_api.py`, update:
```python
LLM_API_URL = "http://your-llm-endpoint/advisory"  # Replace with your API URL
```

## 🏃 Running the Application

### Option 1: Yield Prediction Only
```bash
uvicorn app_blended:app --host 0.0.0.0 --port 8000
```
- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

### Option 2: Full Orchestrator (YOLO + Yield + LLM)
```bash
uvicorn orchestrator_api:app --host 0.0.0.0 --port 8002
```
- API: `http://127.0.0.1:8002`
- Docs: `http://127.0.0.1:8002/docs`

## 🧪 Testing Step-by-Step

### Step 1: Health Check
Verify servers are running:
```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok"}

curl http://127.0.0.1:8002/docs  # Should load FastAPI docs
```

### Step 2: Test Yield Prediction API
Send a POST request to `/predict`:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "district_std": "Anugul",
    "crop_year": 1997,
    "season": "Autumn",
    "area_ha": 948.0,
    "T2M": [24.8, 25.0, 23.7, 24.1],
    "T2M_MAX": [28.8, 29.1, 27.5, 28.2],
    "T2M_MIN": [20.9, 21.3, 19.8, 20.0],
    "PRECTOTCORR": [0.0, 0.0, 5.5, 12.0],
    "RH2M": [83.1, 82.5, 84.0, 85.2],
    "ALLSKY_SFC_SW_DWN": [16.5, 16.9, 15.8, 16.0]
  }'
```

**Expected Response**:
```json
{
  "district_std": "Anugul",
  "crop_year": 1997,
  "season": "Autumn",
  "area_ha": 948.0,
  "pred_benchmark": 0.322,
  "pred_v2": 0.682,
  "pred_v21": 0.756,
  "pred_best_rmse_blend": 0.632,
  "pred_best_mae_blend": 0.438
}
```

### Step 2b: Run YOLO Locally (No Server Required)
If you just want to run the pest/disease detection model directly from your machine, use the provided CLI helper:

```bash
python yolo_cli.py --image path/to/your/crop.jpg --output annotated.jpg
```

This will print a simple diagnosis report and optionally save an annotated image with detections.

### Step 3: Test Full Orchestrator (Image Upload)
Create a `metadata.json` file:
```json
{
  "district_std": "Anugul",
  "crop_year": 1997,
  "season": "Autumn",
  "area_ha": 948.0,
  "T2M": [24.8, 25.0, 23.7, 24.1],
  "T2M_MAX": [28.8, 29.1, 27.5, 28.2],
  "T2M_MIN": [20.9, 21.3, 19.8, 20.0],
  "PRECTOTCORR": [0.0, 0.0, 5.5, 12.0],
  "RH2M": [83.1, 82.5, 84.0, 85.2],
  "ALLSKY_SFC_SW_DWN": [16.5, 16.9, 15.8, 16.0],
  "crop_type": "maize",
  "growth_stage": "vegetative",
  "language": "english"
}
```

Upload an image (use a real crop photo for best results):
```bash
curl -X POST http://127.0.0.1:8002/orchestrate \
  -F "image=@path/to/your/image.jpg" \
  -F "metadata=<metadata.json"
```

**Expected Response**:
```json
{
  "status": "success",
  "visual_diagnosis": {
    "diagnosis": "fall armyworm infestation",
    "confidence": 0.95,
    "severity": "high",
    "annotated_image_base64": "..."
  },
  "environmental_context": {
    "district": "Anugul",
    "season": "Autumn",
    "expected_yield_baseline": "0.63 t/ha",
    "is_error": false
  },
  "expert_advisory": {
    "advisory": "Risk assessment: High severity infestation may reduce yield below 0.63 t/ha. Immediate actions: Apply insecticides... Preventive actions: Crop rotation... Monitoring: Scout weekly... Escalation: Contact extension if spread >20%."
  }
}
```

### Step 4: Web-Based Testing
Open `upload_form.html` in a browser, select an image, adjust metadata if needed, and submit.

## 📁 Project Structure

- `app_blended.py`: Yield prediction API
- `orchestrator_api.py`: Full orchestrator API
- `blended_predictor.py`: Prediction logic
- `*.joblib`: Trained ML models
- `*.csv`: Training/validation data
- `Yolo_pest_identification.pt`: YOLO model
- `upload_form.html`: Simple web form for testing

## 🐛 Troubleshooting

- **Model Loading Errors**: Ensure scikit-learn 1.6.1 is installed.
- **YOLO Fails**: Check image format (JPG/PNG/WEBP); model is classification-based, not detection. Use clear crop images.
- **LLM Timeout**: Verify API URL and network; response may take >30s.
- **Port Conflicts**: If 8001 is occupied, use 8002 as shown above.

## 📞 Support

For issues, check server logs or test individual components first.