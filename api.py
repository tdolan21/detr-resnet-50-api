from transformers import DetrImageProcessor, DetrForObjectDetection
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import random
import torch
import cv2
import io


app = FastAPI()

# Check if CUDA-compatible GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor once during startup and move to the device
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


@app.get("/")
async def root():
    return {"message": "Welcome to the detr-resnet-50 unofficial API. Please use /docs for a list of endpoints with testing."}

app.mount("/download", StaticFiles(directory="download"), name="download")

@app.post("/detect/")
async def detect_objects(file: UploadFile = None):
    # Check if file is uploaded
    if not file:
        raise HTTPException(status_code=400, detail="File not uploaded")

    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Process the image and get predictions
    inputs = processor(images=image, return_tensors="pt")
    # Move input tensors to the device
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    
    outputs = model(**inputs)

    # Convert outputs to COCO API and filter detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Prepare response
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "location": box
        })

    return {"detections": detections}


@app.post("/detect_video/")
async def detect_video(file: UploadFile = None):
    # Check if file is uploaded
    if not file:
        raise HTTPException(status_code=400, detail="File not uploaded")

    # Read the uploaded video file
    video_data = await file.read()
    video_path = f"temp_{file.filename}"
    with open(video_path, 'wb') as f:
        f.write(video_data)

    # Initialize video capture and video writer objects
    vidcap = cv2.VideoCapture(video_path)
    # Get the original frame rate of the video
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_file = "download/output.avi"
    out = cv2.VideoWriter(video_file, fourcc, fps, (int(vidcap.get(3)), int(vidcap.get(4))))
    
    frame_rate = 0.3 # extracts a frame every half-second
    all_detections = []

    success, frame = vidcap.read()
    while success:
        # Convert the frame to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect objects in the frame
        detections = await detect_objects_for_image(pil_img)
        all_detections.extend(detections)

        # Dictionary to hold label-color mappings
        label_color_map = {}

        for detection in detections:
            box = detection["location"]
            label = detection["label"]
            confidence = detection["confidence"]
            
            # Get the color from label_color_map, if not found generate a new color
            color = label_color_map.setdefault(label, generate_random_color())
            
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            thickness = 2
            
            cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video
        out.write(frame)
        # Move to the next frame
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (vidcap.get(cv2.CAP_PROP_POS_MSEC) + (1000 * frame_rate)))
        success, frame = vidcap.read()

    # Release the video objects
    vidcap.release()
    out.release()

    # Provide a URL to download the output video
    video_url = f"/download/{video_file}"

    return JSONResponse(content={
        "video_url": video_url,
        "detections": all_detections
    })

async def detect_objects_for_image(image):
    inputs = processor(images=image, return_tensors="pt")
    for key, value in inputs.items():
        inputs[key] = value.to(device)

    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "location": box
        })

    return detections


# Run using: uvicorn api:app --reload
