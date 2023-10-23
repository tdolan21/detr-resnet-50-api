import streamlit as st
import requests
import subprocess
from PIL import Image, ImageDraw
import imageio
import random
import imageio_ffmpeg as ffmpeg


BASE_API_URL = "http://127.0.0.1:8000/"
ENDPOINTS = {
    "Image Detection": "detect/",
    "Video Detection": "detect_video/"
}
def generate_color():
    """Generate one of the predefined bright colors."""
    colors = ["#C2FF0A", "#FF0A47", "#FF0AC2", "#47FF0A", "#0AC2FF"]
    selected_color = random.choice(colors)
    return tuple(int(selected_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB tuple


def annotate_image(image, detections):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection["location"]
        label = detection["label"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), label, fill="red")
    return image

st.sidebar.title("DETR Object Detection")
selected_endpoint = st.sidebar.selectbox("Select API Endpoint", list(ENDPOINTS.keys()))
API_URL = BASE_API_URL + ENDPOINTS[selected_endpoint]
st.sidebar.divider()

st.title("Object Detection using DETR")

file_type = "jpg" if selected_endpoint == "Image Detection" else ["mp4", "avi"]
uploaded_file = st.sidebar.file_uploader(f"Choose a {file_type}...", type=file_type)
st.sidebar.divider()

if uploaded_file is not None:
    if selected_endpoint == "Image Detection":
        image = Image.open(uploaded_file).convert('RGB')
        st.write("")
        st.info("Detecting...")

        # Send the image to the API
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)
        detections = response.json()["detections"]

        unique_labels = list(set([d['label'] for d in detections]))

        # Check if the color_map exists in the session state
        if 'color_map' not in st.session_state:
            st.session_state.color_map = {}

        # Update the color_map with new labels and their respective colors
        for label in unique_labels:
            if label not in st.session_state.color_map:
                st.session_state.color_map[label] = generate_color()

        # Create checkboxes in the sidebar for each label
        label_selections = {}
        for label in unique_labels:
            label_selections[label] = st.sidebar.checkbox(f"Highlight {label}", value=True)

        # Annotate the image based on the selected labels
        draw = ImageDraw.Draw(image)

        for detection in detections:
            box = detection["location"]
            label = detection["label"]
            confidence = detection['confidence']

            color = st.session_state.color_map[label]
            if label_selections[label]:  # Only annotate if the label's checkbox is selected
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0], box[1]), label, fill=color)

        st.image(image, caption="Detected Objects.", use_column_width=True)
            
        for detection in detections:
            box = detection["location"]
            label = detection["label"]
            confidence = detection['confidence']
            # Display the label, confidence, and location
            st.write(f"Detected {label} with confidence {confidence:.2f} at location {detection['location']}")

        # Calculate average confidence for each unique label
        avg_confidence_per_label = {}
        for label in unique_labels:
            confidences = [d['confidence'] for d in detections if d['label'] == label]
            avg_confidence = sum(confidences) / len(confidences)
            avg_confidence_per_label[label] = avg_confidence

        # Display average confidence for each label
        for label, avg_confidence in avg_confidence_per_label.items():
            st.sidebar.write(f"Average confidence for {label}: {avg_confidence:.2f}")

            # Display a colored progress bar for the average confidence
            color_hex = '#%02x%02x%02x' % st.session_state.color_map[label]
            st.sidebar.markdown(
                f"<div style='background-color: #E0E0E0; border-radius: 10px; height: 20px; position: relative;'>"
                f"<div style='background-color: {color_hex}; border-radius: 10px; height: 100%; width: {avg_confidence * 100}%;'></div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ... [rest of the code for Video Detection]
    elif selected_endpoint == "Video Detection":
        st.write("Detecting objects in video...")
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)
        detections = response.json()["detections"]
        # Convert video to H.264 format for compatibility
        input_video_path = "../download/output.avi"
        output_video_path = "../download/output_h264.mp4"
        
        # Use ffmpeg to convert the video codec
        ffmpeg_cmd = [
        "ffmpeg", "-i", "../download/output.avi",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", 
        "-preset", "fast", "-crf", "24",
        "-y",  # Add this flag to force overwrite
        "../download/output_h264.mp4"
    ]

        result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE)

        if result.returncode != 0:
            st.write("Error executing ffmpeg:", result.stderr.decode())

        # Play the converted video
        st.video(output_video_path)

        # Display all detections
        for detection in detections:
            st.write(f"Detected {detection['label']} with confidence {detection['confidence']} at location {detection['location']}")

        
