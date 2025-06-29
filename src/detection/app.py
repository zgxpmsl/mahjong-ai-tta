from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = FastAPI()

# 加载模型
model_path = os.environ.get('MODEL_PATH', '/models/yolov8-base/best.pt')
model = YOLO(model_path)

@app.get("/")
def read_root():
    return {"message": "麻将检测API服务运行中"}

@app.post("/detect")
async def detect_mahjong(file: UploadFile = File(...)):
    # 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 执行检测
    results = model(img)
    
    # 处理结果
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf),
                    "class": int(box.cls)
                }
                detections.append(detection)
    
    return JSONResponse(content={"detections": detections})

@app.get("/health")
def health_check():
    return {"status": "healthy"}
