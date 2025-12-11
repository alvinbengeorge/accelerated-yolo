import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import hailo_platform as hpf

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources"
VIDEO_PATH = RESOURCE_DIR / "videos" / "example.mp4"
HEF_PATH = RESOURCE_DIR / "yolov11n_h8l.hef"
PT_MODEL_PATH = PROJECT_ROOT / "test" / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Limit frames to avoid extremely long run times if video is long
MAX_FRAMES = 100 

def get_video_frames(video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if count >= max_frames:
            break
    cap.release()
    return frames

def run_regular_yolo(frames):
    print(f"\n--- Running Regular YOLOv11n on {len(frames)} frames ---")
    
    # Check if model exists in test/, else download or use default
    model_path = str(PT_MODEL_PATH) if PT_MODEL_PATH.exists() else "yolo11n.pt"
    print(f"Loading model from: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return []

    inference_times = []
    
    # Warmup
    if len(frames) > 0:
        model(frames[0], verbose=False)

    print("Processing frames...")
    for i, frame in enumerate(frames):
        start_time = time.time()
        results = model(frame, verbose=False)
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        inference_times.append(duration_ms)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames", end='\r')
    
    print(f"\nFinished Regular YOLO. Avg Time: {np.mean(inference_times):.2f} ms")
    return inference_times

def run_hailo_yolo(frames):
    print(f"\n--- Running Hailo YOLOv11n on {len(frames)} frames ---")
    
    if not HEF_PATH.exists():
        print(f"Error: HEF file not found at {HEF_PATH}")
        return []

    inference_times = []

    try:
        hef = hpf.HEF(str(HEF_PATH))
        input_vstream_info = hef.get_input_vstream_infos()[0]
        input_height, input_width = input_vstream_info.shape[0], input_vstream_info.shape[1]

        # Pre-process all frames to avoid counting resize time in inference stats (optional, but fairer for pure inference comparison)
        # However, usually "FPS" includes preprocessing. Let's include preprocessing in the loop but measure pure inference if possible, 
        # or just measure the whole "step" time.
        # The user asked for "average inference time". Usually this means the model execution time.
        # But for "FPS", it implies the throughput.
        # I will measure the specific inference call time for "Inference Time" stats, 
        # and I can calculate FPS based on that or total time.
        # Let's measure the loop time (including resize) for a more realistic FPS, 
        # but also capture the `infer` call duration for "Inference Time".
        
        # Actually, let's stick to measuring the `infer` call for the graph to match the "Inference Time" request strictly,
        # effectively simulating the accelerator speed.
        
        print("Setting up Hailo VDevice...")
        with hpf.VDevice() as target:
            configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]

            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hpf.FormatType.UINT8)
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

            with network_group.activate():
                with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    
                    # Warmup
                    if len(frames) > 0:
                        warmup_img = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), (input_width, input_height))
                        warmup_data = np.expand_dims(warmup_img, axis=0)
                        infer_pipeline.infer({input_vstream_info.name: warmup_data})

                    print("Processing frames...")
                    for i, frame in enumerate(frames):
                        # Preprocessing (included in 'system' FPS, but maybe separate for 'inference' metric)
                        # I'll measure strictly inference time for the graph to show hardware capability.
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        resized_image = cv2.resize(img_rgb, (input_width, input_height))
                        input_data = np.expand_dims(resized_image, axis=0)
                        
                        input_dict = {input_vstream_info.name: input_data}
                        
                        start_time = time.time()
                        results = infer_pipeline.infer(input_dict)
                        end_time = time.time()
                        
                        duration_ms = (end_time - start_time) * 1000
                        inference_times.append(duration_ms)
                        
                        if (i + 1) % 10 == 0:
                            print(f"Processed {i + 1}/{len(frames)} frames", end='\r')

    except Exception as e:
        print(f"Error in Hailo inference: {e}")
        return []

    print(f"\nFinished Hailo YOLO. Avg Time: {np.mean(inference_times):.2f} ms")
    return inference_times

def generate_graph(regular_data, hailo_data):
    print("\nGenerating Graph...")
    plt.figure(figsize=(12, 6))
    
    # Plot Lines
    plt.plot(regular_data, label=f'Regular YOLOv11n (Avg: {np.mean(regular_data):.1f} ms)', color='red', alpha=0.7)
    plt.plot(hailo_data, label=f'Hailo YOLOv11n (Avg: {np.mean(hailo_data):.1f} ms)', color='green', alpha=0.9)
    
    plt.title('Inference Time Comparison: Regular CPU vs Hailo-8L')
    plt.xlabel('Frame Number')
    plt.ylabel('Inference Time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = OUTPUT_DIR / "fps_comparison_graph.png"
    plt.savefig(str(output_path))
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    print(f"Loading video frames from {VIDEO_PATH}...")
    if not VIDEO_PATH.exists():
        print(f"Error: Video file not found at {VIDEO_PATH}")
        exit(1)
        
    frames = get_video_frames(VIDEO_PATH, MAX_FRAMES)
    
    if not frames:
        print("No frames loaded.")
        exit(1)
        
    print(f"Loaded {len(frames)} frames.")
    
    # Run Tests
    reg_times = run_regular_yolo(frames)
    hailo_times = run_hailo_yolo(frames)
    
    # Results
    if reg_times and hailo_times:
        avg_reg = np.mean(reg_times)
        avg_hailo = np.mean(hailo_times)
        
        fps_reg = 1000.0 / avg_reg
        fps_hailo = 1000.0 / avg_hailo
        
        print("\n====== FINAL RESULTS ======")
        print(f"Regular YOLOv11n:")
        print(f"  Avg Inference Time: {avg_reg:.2f} ms")
        print(f"  Est. FPS (Inference): {fps_reg:.2f}")
        
        print(f"Hailo YOLOv11n:")
        print(f"  Avg Inference Time: {avg_hailo:.2f} ms")
        print(f"  Est. FPS (Inference): {fps_hailo:.2f}")
        
        print(f"Speedup Factor: {avg_reg / avg_hailo:.2f}x")
        
        generate_graph(reg_times, hailo_times)
    else:
        print("Tests failed to complete.")
