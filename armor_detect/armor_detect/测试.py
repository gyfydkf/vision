import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# ① 载入模型（CPU 环境请显式 device='cpu'）
model = YOLO('/home/venom/Desktop/armor_detect(1)/armor_detect/runs/detect/train/weights/best.pt')

# ② 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头")

# ③ 调整窗口
cv2.namedWindow("YOLOv8 Armor Detect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Armor Detect", 960, 720)   # 想多大就多大

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ 读取摄像头失败")
        break

    t0 = time.time()

    # ④ 推理 —— 把 *副本* 送进去，避免潜在原地修改
    results = model(frame.copy(), device='cpu', imgsz=320, conf=0.2)


    # ⑤ 在原始帧上手动绘制检测框
    annotator = Annotator(frame)          # 直接用原图
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls   = int(box.cls)
        conf  = float(box.conf)
        label = f"{model.names[cls]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=(0,255,0))  # 绿色框

    vis = annotator.result()

    # ⑥ 显示 FPS
    fps = 1.0 / max(time.time() - t0, 1e-4)
    cv2.putText(vis, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

    # ⑦ Debug 信息（可删）
    # print("mean:", np.mean(vis), "boxes:", len(results[0].boxes))

    cv2.imshow("YOLOv8 Armor Detect", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#
#yolo export model=/home/venom/Desktop/armor_detect(1)/armor_detect/runs/detect/train/weights/best.pt format=openvino imgsz=640