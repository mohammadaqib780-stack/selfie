from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os
import uvicorn

app = FastAPI()

@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        path1 = f"temp_{file1.filename}"
        path2 = f"temp_{file2.filename}"

        # Save uploaded files temporarily
        with open(path1, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # Run DeepFace verification with multiple models
        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace"]
        results = {}

        for m in models:
            try:
                res = DeepFace.verify(path1, path2, model_name=m, enforce_detection=True)
                results[m] = {
                    "verified": res["verified"],
                    "distance": float(res["distance"]),
                }
            except Exception as e:
                results[m] = {"error": str(e)}

        # Final decision: majority vote
        verified_votes = sum(1 for r in results.values() if isinstance(r, dict) and r.get("verified"))
        final_verified = verified_votes >= 3

        return JSONResponse(content={
            "final_verified": final_verified,
            "votes": verified_votes,
            "results": results
        })

    except Exception as e:
        return {"error": str(e)}



