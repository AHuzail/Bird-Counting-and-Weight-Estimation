# Bird Detection & Weight Estimation

A FastAPI service for detecting and counting birds in videos with weight estimation using YOLOv12 and ByteTrack.

## Setup Instructions

1. Install Python 3.7+ (tested with Python 3.11)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the YOLO model file (`yolo12n.pt`) is in the project root directory

## Running the API

Start the FastAPI service:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at `http://localhost:8000/docs`

## API Usage Example

### Using curl:
```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4" \
  -F "fps_sample=1" \
  -F "conf_thresh=0.25" \
  -F "iou_thresh=0.7"
```

### Using the interactive docs:
1. Navigate to `http://localhost:8000/docs`
2. Click on **POST /analyze_video**
3. Click **Try it out**
4. Upload your video file
5. Adjust parameters (optional):
   - `fps_sample`: Frames to sample (default: 1)
   - `conf_thresh`: Confidence threshold (default: 0.25)
   - `iou_thresh`: IOU threshold for NMS (default: 0.7)
6. Click **Execute**

### Response Format:
```json
{
  "counts": [3, 3, 2, ...],
  "weight_estimates": [
    {
      "track_id": 1,
      "weight_estimate": 45.2,
      "confidence": 0.89,
      "uncertainty": 2.1
    }
  ],
  "tracks_sample": [...],
  "output": ["Outputs/annotated_your_video.mp4"]
}
```

## Implementation Details

### Bird Counting Method

**Detection:**
- Uses YOLOv12n (nano) model pre-trained for object detection
- Applies confidence threshold (default 0.25) to filter detections
- Uses Non-Maximum Suppression (NMS) with IOU threshold (default 0.7) to remove duplicate detections

**Tracking:**
- Implements ByteTrack algorithm for multi-object tracking
- Each detected bird is assigned a unique track ID that persists across frames
- The `persist=True` flag maintains track IDs throughout the video
- Frame-by-frame counts are recorded based on unique track IDs present

**Counting Logic:**
- Count = number of unique track IDs present in each frame
- Returns time-series data of bird counts throughout the video

### Weight Estimation Approach

**Method:**
Weight estimation is based on the bounding box area using an allometric scaling relationship:

```
Weight (g) = k × (Area)^1.5
```

Where:
- `k` = calibration constant (0.00005)
- `Area` = bounding box width × height (in pixels²)
- `1.5` = scaling exponent based on biological scaling laws

**Assumptions:**
1. **Fixed Camera Distance**: Assumes consistent distance between camera and birds
2. **Similar Bird Species**: Calibration constant assumes similar body density/morphology
3. **Full Body Visibility**: Assumes complete bird is visible in bounding box
4. **Pixel-to-Size Mapping**: Implicitly maps pixel area to physical area through calibration constant

**Uncertainty Estimation:**
- Calculates standard deviation of weight estimates across all frames for each tracked bird
- Higher uncertainty indicates inconsistent detections or occlusions
- Average confidence from YOLO detections also provided

**Limitations:**
- Requires calibration with known bird weights for accurate estimates
- Accuracy degrades with camera distance variation
- Works best with birds perpendicular to camera view

## Demo Outputs

After running the analysis, you will find:

1. **Annotated Video**: Located in `Outputs/` folder
   - Shows bounding boxes around detected birds
   - Displays track IDs and weight estimates
   - Shows real-time bird count overlay

2. **JSON Response**: Contains:
   - `counts`: Frame-by-frame bird count array
   - `weight_estimates`: Per-bird weight estimates with track IDs
   - `tracks_sample`: Sample tracking data for verification
   - `output`: Path to annotated video file


## Requirements

- Python 3.7+
- FastAPI
- Ultralytics (YOLO)
- OpenCV
- NumPy
