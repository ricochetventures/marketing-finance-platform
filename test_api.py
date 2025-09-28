import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.app import app
import uvicorn

if __name__ == "__main__":
    print("Starting Enhanced Marketing-Finance API on http://localhost:8000")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
