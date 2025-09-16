from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Marketing-Finance Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Marketing-Finance Platform API Running!"}

@app.get("/companies")
async def list_companies():
    return {
        "companies": [
            "Coca-Cola", "PepsiCo", "Nike", "Apple", "Microsoft"
        ]
    }

if __name__ == "__main__":
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
