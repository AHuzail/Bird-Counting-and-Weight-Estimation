# Implementation Details

## Bird Counting Method

### Detection
- **Model**: YOLOv12n (nano) - lightweight YOLO model for object detection
- **Parameters**:
  - `conf_thresh=0.25`: Minimum confidence score to accept detections
  - `iou_thresh=0.7`: Threshold for removing duplicate boxes (NMS)

### Tracking
- **Algorithm**: ByteTrack - assigns unique IDs to each bird across frames
- **Process**: 
  1. Detect birds in each frame using YOLO
  2. Associate detections with existing tracks based on position and appearance
  3. Assign persistent track IDs (1, 2, 3, ...)
  
### Counting
- **Method**: Count unique track IDs per frame
- **Output**: Time-series array of counts `[3, 3, 2, 4, ...]`
- **Advantage**: Prevents double-counting the same bird

---

## Weight Estimation Approach

### Formula
```
Weight (g) = 0.00005 × (BoundingBoxArea)^1.5
```

Where:
- **BoundingBoxArea** = width × height of detection box (in pixels²)
- **1.5** = scaling exponent based on biological allometric laws
- **0.00005** = calibration constant

### Implementation
1. Calculate bounding box area for each detection
2. Apply formula to estimate weight per frame
3. Average weight estimates across all frames for each tracked bird
4. Calculate uncertainty as standard deviation of estimates

### Key Assumptions

**1. Fixed Camera Distance**
- Birds remain at constant distance from camera
- Required for pixel area to accurately represent physical size

**2. Similar Bird Species**
- All birds have similar body density/morphology
- Single calibration constant works for all detections

**3. Full Body Visibility**
- Complete bird visible in bounding box (no occlusions)
- Partial visibility leads to underestimated weights

**4. Calibration Required**
- Constant (0.00005) is generic and requires tuning with known bird weights
- Different cameras/distances need re-calibration

### Limitations
- Inaccurate if birds move closer/farther from camera
- Different species require different calibration
- Occlusions cause weight underestimation
- No pose/orientation compensation

---

## Output Format

### JSON Response
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
  "output": ["Outputs/annotated_video.mp4"]
}
```

### Annotated Video
- Green boxes around detected birds
- Track IDs and weight labels
- Real-time count overlay
