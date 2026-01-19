# CV Text Parser Service

A FastAPI-based service for parsing and extracting information from CV/Resume documents.

## Prerequisites

- Python 3.9+
- pip or conda package manager

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd CVParser
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Project

### Start the FastAPI Server

**With auto-reload (for development):**
```bash
python -m uvicorn main:app --reload
```

**Without auto-reload (for production):**
```bash
python -m uvicorn main:app
```

The server will start on: **http://127.0.0.1:8000**

## Accessing the API

- **Interactive API Documentation (Swagger UI):** http://127.0.0.1:8000/docs
- **Alternative Documentation (ReDoc):** http://127.0.0.1:8000/redoc
- **API Base URL:** http://127.0.0.1:8000

## Project Structure

```
CVParser/
├── main.py              # FastAPI application and endpoints
├── requirements.txt     # Project dependencies
├── index.html          # Web interface
└── README.md           # This file
```

## Features

- Parse CV/Resume text content
- Extract basic information (email, phone, LinkedIn, GitHub)
- Named Entity Recognition (NLP-based)
- Support for PDF and DOCX document formats
- OCR support for scanned documents
- CORS enabled for cross-origin requests

## Notes

- The service requires a working internet connection for the first-time spaCy model download
- The auto-reload feature watches for file changes in the project directory
- Press `CTRL+C` to stop the server
