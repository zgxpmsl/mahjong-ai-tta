import os
import cv2
from ultralytics import YOLO
import json

def process_video():
    # 加载模型
    model_path = os.environ.get('MODEL_PATH', '/models/best.pt')
    model = YOLO(model_path)
    
    # 设置路径
    video_dir = '/data/videos'
    output_dir = os.environ.get('OUTPUT_PATH', '/outputs/processed-frames')
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理视频
    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每30帧处理一次
                if frame_count % 30 == 0:
                    results = model(frame)
                    
                    # 保存检测结果
                    for r in results:
                        if r.boxes is not None:
                            frame_detections = {
                                'frame': frame_count,
                                'detections': []
                            }
                            for box in r.boxes:
                                detection = {
                                    'bbox': box.xyxy[0].tolist(),
                                    'confidence': float(box.conf),
                                    'class': int(box.cls)
                                }
                                frame_detections['detections'].append(detection)
                            
                            detections.append(frame_detections)
                            
                            # 保存带检测框的帧
                            annotated_frame = results[0].plot()
                            frame_path = os.path.join(output_dir, f'{video_file}_frame_{frame_count}.jpg')
                            cv2.imwrite(frame_path, annotated_frame)
                
                frame_count += 1
            
            cap.release()
            
            # 保存检测结果JSON
            json_path = os.path.join(output_dir, f'{video_file}_detections.json')
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
            
            print(f"处理完成: {video_file}, 共{frame_count}帧")

if __name__ == "__main__":
    process_video()
