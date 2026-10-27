"""Full-pipeline CPU latency (paper, Table 8): BlazeFace detector + crop/resize (student side) and
MediaPipe FaceLandmarker with blendshapes (teacher side), median per processed frame.

    python scripts/pipeline_latency.py data/raw/Actor_01/01-*.mp4
"""
import sys, time, json, urllib.request
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
cv2.setNumThreads(1)
LM_PATH, FD_PATH = 'face_landmarker.task', 'blaze_face_short_range.tflite'
LM = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
FD = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
for u, f in ((LM, LM_PATH), (FD, FD_PATH)):
    try: open(f, 'rb').close()
    except OSError: urllib.request.urlretrieve(u, f)
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=LM_PATH), running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True, num_faces=1))
fd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=mpp.BaseOptions(model_asset_path=FD_PATH), running_mode=vision.RunningMode.IMAGE))
t_lm, t_fd, t_crop, n, found = [], [], [], 0, 0
for path in sys.argv[1:]:
    cap = cv2.VideoCapture(path); i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i % 2 == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t = time.perf_counter(); r1 = lm.detect(img); t_lm.append(time.perf_counter() - t)
            t = time.perf_counter(); r2 = fd.detect(img); t_fd.append(time.perf_counter() - t)
            if r2.detections:
                b = r2.detections[0].bounding_box; found += 1
                t = time.perf_counter()
                cx, cy, s = b.origin_x + b.width / 2, b.origin_y + b.height / 2, max(b.width, b.height) * 1.25
                h, w = rgb.shape[:2]
                crop = rgb[int(max(cy - s / 2, 0)):int(min(cy + s / 2, h)), int(max(cx - s / 2, 0)):int(min(cx + s / 2, w))]
                crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                t_crop.append(time.perf_counter() - t)
            n += 1
        i += 1
    cap.release()
ms = lambda v: 1000 * float(np.median(v[5:])) if len(v) > 5 else float('nan')
res = dict(frames=n, face_found=found, landmarker_ms=ms(t_lm), face_detector_ms=ms(t_fd), crop_resize_ms=ms(t_crop),
           input_resolution=f"{int(rgb.shape[1])}x{int(rgb.shape[0])}", note='median per processed frame, CPU, 1 thread for OpenCV')
json.dump(res, open('results/pipeline_latency.json', 'w'), indent=1); print(res)
