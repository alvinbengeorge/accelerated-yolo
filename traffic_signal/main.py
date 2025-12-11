import cv2
import numpy as np
import time
import os
from pathlib import Path
import hailo_platform as hpf
import datetime
from scipy.spatial import distance as dist
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources"
HEF_PATH = RESOURCE_DIR / "yolov11n_h8l.hef"
VIDEO_PATH = Path(__file__).parent / "traffic_video_2.mp4"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

COLORS = {
    2: (0, 255, 0),
    3: (255, 0, 0),
    5: (0, 255, 255),
    7: (0, 0, 255)
}

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

class SpeedEstimator:
    def __init__(self, pos1_y, pos2_y, distance_meters, fps):
        self.line1_y = pos1_y
        self.line2_y = pos2_y
        self.distance = distance_meters
        self.fps = fps
        # entry_data: {objectID: (frame_num, line_index)} 
        # line_index: 1 for line1, 2 for line2
        self.entry_data = {} 
        self.speeds = {} 
        self.previous_centroids = {}

    def update(self, objects, frame_count):
        for objectID, centroid in objects.items():
            cy = centroid[1]
            if objectID in self.previous_centroids:
                prev_cy = self.previous_centroids[objectID][1]
                
                # Check Line 1 Crossing
                if (prev_cy < self.line1_y <= cy) or (prev_cy > self.line1_y >= cy):
                    self.handle_crossing(objectID, frame_count, 1)

                # Check Line 2 Crossing
                if (prev_cy < self.line2_y <= cy) or (prev_cy > self.line2_y >= cy):
                    self.handle_crossing(objectID, frame_count, 2)

            self.previous_centroids[objectID] = centroid
        return self.speeds

    def handle_crossing(self, objectID, frame_num, line_idx):
        if objectID in self.entry_data:
            start_frame, start_line = self.entry_data[objectID]
            if start_line != line_idx:
                # Crossed the other line!
                frames = abs(frame_num - start_frame)
                if frames > 0:
                    time_s = frames / self.fps
                    speed_mps = self.distance / time_s
                    speed_kmh = speed_mps * 3.6
                    self.speeds[objectID] = speed_kmh
        else:
            # First line crossed
            self.entry_data[objectID] = (frame_num, line_idx)

def draw_overlay(image, fps, vehicle_count):
    h, w = image.shape[:2]
    
    # Top Bar Background
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Camera Info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f"CAM-01 | LIVE | {timestamp}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Stats
    stats_text = f"FPS: {fps:.1f} | Vehicles: {vehicle_count}"
    (tw, th), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(image, stats_text, (w - tw - 20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Recording Indicator (Blinking Red Dot)
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(image, (w - 25, 80), 8, (0, 0, 255), -1)

def main():
    print("Initializing Traffic Camera...")
    
    if not HEF_PATH.exists():
        print(f"Error: HEF file not found at {HEF_PATH}")
        return

    if not VIDEO_PATH.exists():
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # Load Hailo HEF
    hef = hpf.HEF(str(HEF_PATH))
    input_vstream_info = hef.get_input_vstream_infos()[0]
    input_height, input_width = input_vstream_info.shape[0], input_vstream_info.shape[1]

    # Open Video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Resolution: {orig_w}x{orig_h}")
    print("Press 'q' to quit.")

    # Output Directory
    SNAPSHOT_DIR = Path("snapshots")
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    
    # Video Writer Setup
    output_video_path = Path(__file__).parent / "traffic_inference.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps_out, (orig_w, orig_h))
    print(f"Exporting video to {output_video_path}...")

    # Initialize Tracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=100)

    # Initialize Speed Estimator
    # lines at 30% and 70% of height, assuming 20 meters distance
    line1_y = int(orig_h * 0.3)
    line2_y = int(orig_h * 0.7)
    speed_estimator = SpeedEstimator(line1_y, line2_y, distance_meters=40, fps=fps_out)
    
    # Track history for drawing lines
    track_history = {} # Format: {objectID: [(x, y), ...]}
    max_trail_length = 30

    # Setup Hailo
    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        with network_group.activate():
            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                

                fps_start_time = time.time()
                frame_count = 0
                fps = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video.")
                        break

                    # Preprocessing
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(img_rgb, (input_width, input_height))
                    input_data = np.ascontiguousarray(resized_image, dtype=np.uint8)
                    input_data = np.expand_dims(input_data, axis=0)

                    # Inference
                    input_dict = {input_vstream_info.name: input_data}
                    results = infer_pipeline.infer(input_dict)
                    
                    # Post-processing
                    raw_data = None
                    for key, value in results.items():
                        if 'nms_postprocess' in key:
                            raw_data = value[0]
                            break
                    
                    rects = [] 
                    detected_boxes_this_frame = [] 

                    if raw_data is not None:
                        for class_id, detections in enumerate(raw_data):
                            if class_id in VEHICLE_CLASSES and detections.shape[0] > 0:
                                for i in range(detections.shape[0]):
                                    score = detections[i][4]
                                    if score > 0.5: 
                                        ymin, xmin, ymax, xmax = detections[i][:4]
                                        
                                        x1 = int(xmin * orig_w)
                                        y1 = int(ymin * orig_h)
                                        x2 = int(xmax * orig_w)
                                        y2 = int(ymax * orig_h)
                                        
                                        rects.append((x1, y1, x2, y2))
                                        
                                        label_text = VEHICLE_CLASSES[class_id]
                                        color = COLORS.get(class_id, (255, 255, 255))
                                        
                                        detected_boxes_this_frame.append({
                                            'box': (x1, y1, x2, y2),
                                            'label': label_text,
                                            'color': color,
                                            'score': score
                                        })

                    # Update Tracker
                    objects = ct.update(rects)

                    # Update Speed Estimator
                    current_speeds = speed_estimator.update(objects, frame_count)
                    
                    # Draw Speed Lines
                    cv2.line(frame, (0, line1_y), (orig_w, line1_y), (0, 255, 255), 2)
                    cv2.line(frame, (0, line2_y), (orig_w, line2_y), (0, 255, 255), 2)
                    cv2.putText(frame, "Line 1", (10, line1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, "Line 2", (10, line2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Update track history
                    for objectID, centroid in objects.items():
                        if objectID not in track_history:
                            track_history[objectID] = []
                        
                        # Add new centroid
                        track_history[objectID].append(centroid)
                        
                        # Limit history length
                        if len(track_history[objectID]) > max_trail_length:
                            track_history[objectID].pop(0)
                            
                    # Clean up history for deregistered objects
                    current_ids = set(objects.keys())
                    track_history = {k: v for k, v in track_history.items() if k in current_ids or (k in ct.disappeared and ct.disappeared[k] < ct.maxDisappeared)}

                    # Dynamic visual scaling
                    resolution_scale = min(orig_w, orig_h) / 1000.0
                    box_thickness = max(1, int(2 * resolution_scale))
                    font_scale = max(0.4, 0.5 * resolution_scale)
                    font_thickness = max(1, int(1 * resolution_scale))
                    line_thickness = max(1, int(2 * resolution_scale))

                    # Draw trails first (so they are under boxes)
                    for objectID, points in track_history.items():
                        if len(points) > 1:
                            draw_color = (0, 255, 255) # Default Yellowish for trails
                            
                            # Try to find matching box color from current frame
                            centroid = points[-1]
                            for item in detected_boxes_this_frame:
                                (bx1, by1, bx2, by2) = item['box']
                                if bx1 <= centroid[0] <= bx2 and by1 <= centroid[1] <= by2:
                                    draw_color = item['color']
                                    break
                            
                            for j in range(1, len(points)):
                                if points[j - 1] is None or points[j] is None:
                                    continue
                                cv2.line(frame, (points[j - 1][0], points[j - 1][1]), (points[j][0], points[j][1]), draw_color, line_thickness)

                    # Draw boxes and labels
                    for item in detected_boxes_this_frame:
                        (x1, y1, x2, y2) = item['box']
                        label_text = item['label']
                        color = item['color']
                        
                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                        
                        # Match this box to a tracked object ID based on centroid proximity
                        cX = int((x1 + x2) / 2.0)
                        cY = int((y1 + y2) / 2.0)
                        
                        matched_id = None
                        min_dist = 99999
                        
                        for (objectID, centroid) in objects.items():
                            d = np.sqrt((cX - centroid[0])**2 + (cY - centroid[1])**2)
                            if d < min_dist and d < 50: # Threshold to visually link
                                min_dist = d
                                matched_id = objectID
                        
                        # id_text = f"ID: {matched_id}" if matched_id is not None else ""
                        # display_label = f"{label_text} {id_text}"
                        display_label = label_text # Just the class name

                        if matched_id is not None and matched_id in current_speeds:
                            speed = current_speeds[matched_id]
                            display_label += f" {speed:.0f} km/h"
                        
                        # Draw Label
                        (w_text, h_text), baseline = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        cv2.rectangle(frame, (x1, y1), (x1 + w_text + 4, y1 + h_text + 6), color, -1)
                        cv2.putText(frame, display_label, (x1 + 2, y1 + h_text + 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                        
                        # Print log
                        print(f"  Frame {frame_count}: Detected {display_label} (Conf: {item['score']:.2f})")


                    # FPS Calculation
                    frame_count += 1
                    if frame_count % 10 == 0:
                        curr_time = time.time()
                        fps = 10 / (curr_time - fps_start_time)
                        fps_start_time = curr_time

                    # Overlay
                    draw_overlay(frame, fps, len(detected_boxes_this_frame))
                    
                    # Save every frame to snapshots folder
                    # snapshot_path = SNAPSHOT_DIR / f"frame_{frame_count:04d}.jpg"
                    # cv2.imwrite(str(snapshot_path), frame)
                    
                    # Write to video
                    video_writer.write(frame)
                    
                    # Print progress
                    if frame_count % 50 == 0:
                        print(f"Processed {frame_count} frames...", end='\r')

    cap.release()
    video_writer.release()

    print(f"\nTraffic Camera Stopped. Total frames processed: {frame_count}")
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
