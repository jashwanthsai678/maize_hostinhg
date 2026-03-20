# 🚀 Step-by-Step Running Instructions

Follow these steps to get your Intelligent Maize Advisory System running and ready for your demo.

## Phase 1: Preparation

1.  **Install Dependencies**:
    Open a terminal in your project folder and run:
    ```bash
    pip install fastapi uvicorn ultralytics pillow requests httpx python-multipart python-dotenv joblib pandas numpy
    ```

2.  **Configure Environment**:
    Open the `.env` file in your project directory and update the following:
    - `LLM_API_URL`: Set this to your VM's IP/URL (e.g., `http://136.112.81.111:8000/advisory`).

3.  **Check Model Paths**:
    Ensure `Yolo_pest_identification.pt` and your `.joblib` model files are in the same folder as the scripts.

## Phase 2: Launching the System

1.  **Run the Demo Script**:
    Double-click the `run_demo.bat` file. This script will:
    - Start the **YOLO Inference Service** on port `8002`.
    - Start the **Yield Prediction Service** on port `8003`.
    - Start the **Advisory Orchestrator** on port `8001`.
    - Automatically open `index.html` in your web browser.

2.  **Wait for Initialization**:
    Wait about 5-10 seconds for the YOLO model to load in the background (you'll see a success message in one of the command windows).

## Phase 3: Using the Interface

1.  **Upload an Image**:
    Drag and drop a crop image (like `pest2.webp`) onto the upload area or click "Browse" to select one.
2.  **Set Context**:
    Click on the **Environmental Context** header to expand it and adjust the district, season, or language if needed.
3.  **Generate Advisory**:
    Click the **Generate Advisory** button.
4.  **Review Results**:
    - **Visual Diagnosis**: View the annotated image and detected issue.
    - **Yield Projection**: See the expected yield baseline for the selected district.
    - **Expert Advisory**: Read the AI-generated agronomic advice tailored to your specific case.

## Phase 4: Stopping the System

1.  Close the browser tab.
2.  Find the main command prompt window opened by `run_demo.bat` and press any key to stop all background services automatically.
