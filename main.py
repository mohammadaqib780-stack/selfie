from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import os

app = FastAPI()

@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        path1 = f"temp_{file1.filename}"
        path2 = f"temp_{file2.filename}"

        # Save temp files
        with open(path1, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # Use only reliable models
        models = ["ArcFace", "Facenet"]
        THRESHOLD = 0.85  # More flexible (default is 0.4 → too strict)

        results = {}
        votes = 0

        for m in models:
            try:
                res = DeepFace.verify(
                    path1,
                    path2,
                    model_name=m,
                    enforce_detection=False  # Disable strict detection
                )

                distance = float(res["distance"])
                verified = distance < THRESHOLD

                if verified:
                    votes += 1

                results[m] = {
                    "verified": verified,
                    "distance": distance
                }

            except Exception as e:
                results[m] = {"error": str(e)}

        # FINAL DECISION (soft rule)
        final_verified = votes >= 1  # at least 1 model matches

        # Delete temp files
        os.remove(path1)
        os.remove(path2)

        return JSONResponse(content={
            "final_verified": final_verified,
            "votes": votes,
            "results": results
        })

    except Exception as e:
        return {"error": str(e)}
