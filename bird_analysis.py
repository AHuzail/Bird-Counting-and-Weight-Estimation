import cv2
import numpy as np
from ultralytics import YOLO
import os

class BirdAnalyzer:
    def __init__(self, model_name='yolo12n.pt'):

        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.calibration_constant_k = 0.00005

    def process_video(self, video_path, output_path, conf_thresh=0.25, iou_thresh=0.7, fps_sample=1):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Data collection
        counts = []
        weight_estimates = [] # List of dicts
        tracks_sample = [] # Small sample of tracks
        
        frame_idx = 0
        
        track_history = {} # track_id -> {'areas': [], 'confs': []}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Tracking
            # Persist=True is important for tracking
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                       conf=conf_thresh, iou=iou_thresh, verbose=False)
            
            result = results[0]
            current_count = 0
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                current_count = len(np.unique(track_ids))
                
                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    
                    # Weight estimation
                    est_weight = self.calibration_constant_k * (area ** 1.5)
                    
                    # Update history
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append({'weight': est_weight, 'conf': conf})

                    # Annotation
                    weight_text = f"{est_weight:.1f}g"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{track_id} {weight_text}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                    # Collect sample tracks (e.g., from first few frames or periodically)
                    if frame_idx % 30 == 0: # Every ~1 second
                         tracks_sample.append({
                             "frame_idx": frame_idx,
                             "id": int(track_id),
                             "box": [float(x1), float(y1), float(x2), float(y2)]
                         })

            # Overlay count
            cv2.putText(frame, f"Count: {current_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(frame)
            
            # Store counts
            counts.append(int(current_count))
            frame_idx += 1

        cap.release()
        out.release()
        
        # Summarize weights per track
        final_weights = []
        for tid, history in track_history.items():
            avg_weight = np.mean([h['weight'] for h in history])
            avg_conf = np.mean([h['conf'] for h in history])
            # simple uncertainty proxy: std dev of weight estimates
            uncertainty = np.std([h['weight'] for h in history]) if len(history) > 1 else 0.0
            
            final_weights.append({
                "track_id": int(tid),
                "weight_estimate": float(avg_weight),
                "confidence": float(avg_conf),
                "uncertainty": float(uncertainty)
            })

        return {
            "counts": counts, # Time series of counts
            "tracks_sample": tracks_sample,
            "weight_estimates": final_weights,
            "artifacts": [output_path]
        }
