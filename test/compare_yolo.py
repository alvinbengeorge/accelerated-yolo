import cv2
import numpy as np
import time
import os
from pathlib import Path
from ultralytics import YOLO
import hailo_platform as hpf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources"
HEF_PATH = RESOURCE_DIR / "yolov11n_h8l.hef"
IMAGE_PATH = PROJECT_ROOT / "test_images" / "test.jpg"
OUTPUT_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR.mkdir(exist_ok=True)

# COCO Labels
COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 
    78: "hair drier", 79: "toothbrush"
}

def run_regular_yolo():
    print("\n--- Running Regular YOLOv11n (Ultralytics) ---")
    
    # Load model
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Error loading YOLOv11n model: {e}")
        return

    # Read image
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        print(f"Error: Could not read image {IMAGE_PATH}")
        return

    # Warmup
    print("Warming up...")
    model(img, verbose=False)

    # Inference
    print("Starting inference...")
    start_time = time.time()
    results = model(img, verbose=False)
    end_time = time.time()
    
    duration = (end_time - start_time) * 1000
    print(f"Regular YOLO Inference Time: {duration:.2f} ms")

    # Save result
    res_plotted = results[0].plot()
    cv2.imwrite(str(OUTPUT_DIR / "regular_yolo_result.jpg"), res_plotted)
    print(f"Saved result to {OUTPUT_DIR / 'regular_yolo_result.jpg'}")
    
    return duration

def run_hailo_yolo():
    print("\n--- Running Hailo YOLOv11n ---")


    if not HEF_PATH.exists():
        print(f"Error: HEF file not found at {HEF_PATH}")
        return

    try:
        # Load HEF
        hef = hpf.HEF(str(HEF_PATH))
        input_vstream_info = hef.get_input_vstream_infos()[0]
        
        # Read image
        img_bgr = cv2.imread(str(IMAGE_PATH))
        if img_bgr is None:
            print(f"Error: Could not read image {IMAGE_PATH}")
            return

        # Preprocessing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_height, input_width = input_vstream_info.shape[0], input_vstream_info.shape[1]
        resized_image = cv2.resize(img_rgb, (input_width, input_height))
        input_data = np.expand_dims(resized_image, axis=0)

        # Inference
        print("Starting inference...")
        
        # VDevice setup
        with hpf.VDevice() as target:
            configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]

            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

            with network_group.activate():
                with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    # Warmup
                    input_dict = {input_vstream_info.name: input_data}
                    infer_pipeline.infer(input_dict)
                    
                    # Time measurement
                    start_time = time.time()
                    results = infer_pipeline.infer(input_dict)
                    end_time = time.time()

        duration = (end_time - start_time) * 1000
        print(f"Hailo Inference Time: {duration:.2f} ms")

        # Post-processing
        output_image = cv2.resize(img_bgr, (input_width, input_height)) # Draw on resized or original? Let's use resized to match bounding box scale logic easily
        # Or better, scale boxes back to original image size if we want accurate comparison. 
        # The regular YOLO output is on original image size.
        # Let's scale boxes to original image size.
        
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Find the correct output key. It might be 'yolov8m/yolov8_nms_postprocess' or similar.
        # We'll search for it.
        raw_data = None
        for key, value in results.items():
            if 'nms_postprocess' in key:
                raw_data = value[0] # Batch 0
                break
        
        if raw_data is None:
            # Fallback or error
            print("Could not find NMS postprocess output in Hailo results. Keys:", results.keys())
            return duration

        detection_count = 0
        for class_id, detections in enumerate(raw_data):
            if detections.shape[0] > 0:
                for i in range(detections.shape[0]):
                    score = detections[i][4]
                    if score > 0.5:
                        ymin, xmin, ymax, xmax = detections[i][:4]
                        
                        # Scale to ORIGINAL image dimensions
                        x1 = int(xmin * orig_w)
                        y1 = int(ymin * orig_h)
                        x2 = int(xmax * orig_w)
                        y2 = int(ymax * orig_h)
                        
                        label = COCO_LABELS.get(class_id, f"Class {class_id}")
                        caption = f"{label} {score:.2f}"
                        
                        # Draw box
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                        cv2.putText(img_bgr, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        detection_count += 1

        print(f"Hailo Detections found: {detection_count}")
        
        cv2.imwrite(str(OUTPUT_DIR / "hailo_yolo_result.jpg"), img_bgr)
        print(f"Saved result to {OUTPUT_DIR / 'hailo_yolo_result.jpg'}")
        
        return duration

    except Exception as e:
        print(f"Error in Hailo inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    t_reg = run_regular_yolo()
    t_hailo = run_hailo_yolo()
    
    print("\n--- Summary ---")
    if t_reg:
        print(f"Regular YOLOv11n: {t_reg:.2f} ms")
    if t_hailo:
        print(f"Hailo YOLOv11n:   {t_hailo:.2f} ms")
    
    if t_reg and t_hailo:
        speedup = t_reg / t_hailo
        print(f"Speedup: {speedup:.2f}x")
