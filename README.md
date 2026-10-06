# AutoChecker

## Overview
AutoChecker is an automated grading and evaluation system for subjective exam answer sheets. It processes handwritten PDF answer sheets by:
- Correcting scan skew and tilt using OpenCV Hough Line transforms.
- Segmenting individual lines of handwritten text into discrete image segments.
- Extracting handwritten text using Google Cloud Vision Document Text Detection OCR.
- Segmenting diagrams and scoring them against an answer key diagram using OpenCLIP (`ViT-B-32`).
- Computing semantic similarity between student answers and the text key using SentenceTransformer (`stsb-roberta-large`).
- Presenting an interactive web dashboard with configurable weighting between textual and diagrammatic marks.

### Features
- **Handwritten OCR**: Extracts text from handwritten answer sheets using Google Cloud Vision API.
- **Line Segmentation & Deskewing**: Automatically straightens tilted scans and extracts clean line crops.
- **Diagram Similarity**: Detects diagrams and compares them against answer key reference images via OpenCLIP embeddings.
- **Semantic Text Matching**: Evaluates subjective answers conceptually using transformer embeddings rather than exact keyword matching.
- **Interactive UI**: Upload multiple student PDFs simultaneously, tune text vs. diagram scoring weights, and inspect extracted segments.

---

## Prerequisites

1. **Python 3.10+** (Python 3.12 recommended)
2. **Node.js 18+** & npm
3. **Poppler** (Required by `pdf2image` for PDF rendering):
   - **Linux/Docker**: `apt-get install -y poppler-utils`
   - **macOS**: `brew install poppler`
   - **Windows**: Install via `winget install -e --id osdn.poppler` or download poppler for Windows and add the `bin/` directory to your system `PATH`.
4. **Google Cloud Vision API Credentials**: A valid Google Cloud service account with Vision API enabled.

---

## Google Cloud Vision Setup

1. In the [Google Cloud Console](https://console.cloud.google.com/), select your project and enable the **Cloud Vision API**.
2. Go to **IAM & Admin** > **Service Accounts** and create a service account with the role **Cloud Vision API User**.
3. Create and download a service account JSON key file.
4. Set the environment variable pointing to your downloaded key file:
   - **Linux / macOS**:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
     ```
   - **Windows (PowerShell)**:
     ```powershell
     $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
     ```
   > ⚠️ **Security Warning:** Never commit your service account JSON file to GitHub or any public repository. Keep it outside the project directory or listed in `.gitignore`.

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/platinumcore376/Auto-checker.git
cd Auto-checker
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# Activate the virtual environment:
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Start the backend server:
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## Docker Setup
You can also run both the backend and frontend using Docker Compose:
```bash
docker compose up --build
```
Ensure your Google Cloud credentials are provided as an environment variable or volume mount when running in production.

---

## Usage
1. Enter the textual answer key in the left panel.
2. Upload the reference diagram image for the answer key.
3. Configure the Total Marks and adjust Text Weight / Diagram Weight (weights must sum to 1.0).
4. Click **Add Answer Sheets+** to upload student PDF answer sheets.
5. Click **AutoCorrect** to process all submissions.
6. Click any student card to inspect the extracted text crop, extracted diagram, and similarity scores.
