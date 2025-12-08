from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil, os

app = FastAPI()

@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        path1 = f"temp_{file1.filename}"
        path2 = f"temp_{file2.filename}"

        with open(path1, "wb") as buffer:
            shutil.copyfileobj(file1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        # -------- STRICT FACE DETECTION (MTCNN) --------
        def has_valid_face(img_path):
            try:
                faces = DeepFace.extract_faces(img_path, detector_backend='mtcnn')
                if len(faces) == 0:
                    return False
                if faces[0]["confidence"] < 0.90:
                    return False
                return True
            except:
                return False

        if not has_valid_face(path1) or not has_valid_face(path2):
            os.remove(path1); os.remove(path2)
            return JSONResponse(content={
                "final_verified": False,
                "error": "No real human face detected."
            })

        # -------- FACE VERIFICATION (MTCNN + ArcFace/Facenet) --------
        models = ["ArcFace", "Facenet"]
        THRESHOLD = 0.85

        results = {}
        votes = 0

        for m in models:
            res = DeepFace.verify(
                img1_path=path1,
                img2_path=path2,
                model_name=m,
                detector_backend="mtcnn",
                enforce_detection=True
            )

            distance = float(res["distance"])
            verified = distance < THRESHOLD

            if verified:
                votes += 1

            results[m] = {"verified": verified, "distance": distance}

        final_verified = votes >= 1

        os.remove(path1)
        os.remove(path2)

        return JSONResponse(content={
            "final_verified": final_verified,
            "votes": votes,
            "results": results
        })

    except Exception as e:
        return {"error": str(e)}
