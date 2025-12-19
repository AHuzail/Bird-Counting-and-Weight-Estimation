from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import tempfile
from bird_analysis import BirdAnalyzer

app = FastAPI()

# Temporary storage for uploaded files and artifacts
UPLOAD_DIR = "uploads"
ARTIFACTS_DIR = "artifacts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


try:
    analyzer = BirdAnalyzer(model_name='yolo12n.pt')
except Exception as e:
    print(f"Warning: Failed to load BirdAnalyzer at startup: {e}")
    analyzer = None

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: int = Form(1),
    conf_thresh: float = Form(0.25),
    iou_thresh: float = Form(0.7)
):
    global analyzer
    if analyzer is None:
        # Retry loading if it failed initially (e.g. download issue)
        analyzer = BirdAnalyzer(model_name='yolo12n.pt')

    # Save uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Define output path
    output_filename = f"annotated_{file.filename}"
    output_path = os.path.join(ARTIFACTS_DIR, output_filename)
    
    try:
        results = analyzer.process_video(
            video_path=file_location, 
            output_path=output_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            fps_sample=fps_sample
        )
        
        # Return results
        return JSONResponse(content=results)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Cleanup input file? Maybe keep for debugging. 
        # For this minimal service, we'll keep it.
        pass

if __name__ == "__main__":
    uvicorn.run(app)
