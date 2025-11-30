from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from pathlib import Path
import hailo_platform as hpf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Hailo Model Configuration ---
# We are using the model compatible with HAILO8L architecture.
HEF_PATH = Path("../resources/yolov8m_h8l.hef")

if not HEF_PATH.exists():
    raise FileNotFoundError(f"HEF file not found at {HEF_PATH}. Please run the download_resources.sh script from the project root.")

# Load HEF and get input/output stream information
hef = hpf.HEF(str(HEF_PATH))
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_infos = hef.get_output_vstream_infos()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content="""
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/static/index.html">
        </head>
        <body>
            <p>Redirecting to the main page...</p>
        </body>
    </html>
    """)

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    try:
        # --- Pre-processing ---
        np_image = np.frombuffer(image_data, np.uint8)
        cv_image_bgr = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # The model expects RGB, BGR is the default for cv2.
        cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

        # Resize the image to the model's expected input size
        input_height, input_width = input_vstream_info.shape[0], input_vstream_info.shape[1]
        resized_image = cv2.resize(cv_image_rgb, (input_width, input_height))

        # The model expects a 3D tensor (H, W, C).
        # We use UINT8 input for the accelerator, so no normalization is needed if we configure the VStream correctly.
        input_data = np.expand_dims(resized_image, axis=0)

        # --- Hailo Inference ---
        with hpf.VDevice() as target:
            configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]

            # Configure input as UINT8 (quantized=True matches the model expectation)
            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

            with network_group.activate():
                with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    input_dict = {input_vstream_info.name: input_data}
                    results = infer_pipeline.infer(input_dict)

        # --- Post-processing ---
        # The model returns a list of 80 arrays (one per class), each with shape (N, 5)
        # Each detection is [ymin, xmin, ymax, xmax, score] (normalized 0-1)
        
        # COCO Labels (subset for demo purposes, ideally load full list)
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

        output_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        raw_data = results['yolov8m/yolov8_nms_postprocess'][0] # Batch 0
        
        for class_id, detections in enumerate(raw_data):
            if detections.shape[0] > 0:
                for i in range(detections.shape[0]):
                    score = detections[i][4]
                    if score > 0.5:
                        ymin, xmin, ymax, xmax = detections[i][:4]
                        
                        # Scale to image dimensions
                        x1 = int(xmin * input_width)
                        y1 = int(ymin * input_height)
                        x2 = int(xmax * input_width)
                        y2 = int(ymax * input_height)
                        
                        label = COCO_LABELS.get(class_id, f"Class {class_id}")
                        caption = f"{label} {score:.2f}"
                        
                        # Draw box
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label background
                        (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(output_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(output_image, caption, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(output_image, "Hailo Inference Complete", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode to PNG
        res, im_png = cv2.imencode(".png", output_image)
        return Response(content=im_png.tobytes(), media_type="image/png")

    except Exception as e:
        return Response(content=str(e), status_code=500)