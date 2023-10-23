# Object Detection API using DETR

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

This project is an object detection API built using FastAPI and Streamlit. It utilizes the DETR (DEtection TRansformer) model for end-to-end object detection.

The detr-resnet-50 model is trained on the COCO image dataset.

## Usage

- Clone the repository to your machine:
```bash
git clone https://github.com/tdolan21/detr-resnet-50-api.git
cd detr-resnet-50-api
pip install -r requirements.txt
```
- Windows machines can run the code by using the batch script:

```bash
 ./run.bat
```
- Linux machines can use the bash script:

```bash
chmod +x run.sh
./run.sh
```

Once you have run the startup script the application will be available at:

```
localhost:8000
localhost:8501
```

## Key Features:
- **Object Detection in Images**: Detect and label objects in uploaded images.
- **Object Detection in Videos**: Detect, label, and annotate objects frame by frame in uploaded videos.
- **Fast, Accurate & Scalable**: Utilizes the DETR model for precise object detection.
- **Interactive**: Offers colorful visualization for detected objects.

## Endpoints:

1. **`/` (GET)**
    - Welcomes users and provides a brief introduction to the API.
    
2. **`/detect/` (POST)**
    - Accepts an image file.
    - Detects and labels objects in the image.
    - Returns a list of detected objects with their labels, confidence scores, and bounding box coordinates.
    
3. **`/detect_video/` (POST)**
    - Accepts a video file.
    - Detects and labels objects frame by frame in the video.
    - Provides an output video with annotated objects.
    - Returns a URL for the annotated video download and a list of detections for each frame.

## Usage:

Simply upload an image or video through the respective endpoint and receive annotated results with detected objects.


## Citation

```
@article{DBLP:journals/corr/abs-2005-12872,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  journal   = {CoRR},
  volume    = {abs/2005.12872},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.12872},
  archivePrefix = {arXiv},
  eprint    = {2005.12872},
  timestamp = {Thu, 28 May 2020 17:38:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-12872.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
