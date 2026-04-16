# Recommendation App

A FastAPI-based recommendation service that uses sentence embeddings + FAISS and Groq for query normalization/translation.

## 1. Create and Activate Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 3. Set Groq API Key in Terminal

Set it in the same terminal session where you will run the server:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Optional quick check:

```powershell
echo $env:GROQ_API_KEY
```

## 4. Run the API

```powershell
python -m uvicorn app:app --reload
```

Server will start at:

- http://127.0.0.1:8000

## 5. Test Endpoints

Home:

- http://127.0.0.1:8000/

Recommendation example:

- http://127.0.0.1:8000/recommend?q=joote%20hai%20apke%20pass%3F

## Notes

- The app logs original query and English query used for search in terminal.
- Keep API keys out of source files. Use environment variables only.

## Troubleshooting: "Failed to fetch"

If browser clients show:

- `Failed to fetch`
- `URL scheme must be "http" or "https" for CORS request`

Check the following:

- Always call the API with a full URL, for example:
	- `https://recommendation-app-49z7.onrender.com/recommend?q=shoes`
- Do not use URLs like `recommendation-app-49z7.onrender.com/recommend?...` (missing scheme).
- If you want restricted CORS, set `CORS_ALLOWED_ORIGINS` as a comma-separated env var:
	- `https://your-frontend.com,https://www.your-frontend.com`
