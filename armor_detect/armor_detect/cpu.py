import cv2, time, numpy as np
from openvino.runtime import Core
from ultralytics.utils.plotting import Annotator

IMG_SIZE = 640
model_path = "/home/venom/Desktop/armor_detect(1)/armor_detect/runs/detect/train/weights/best_openvino_model/best.xml"

# ---------- OpenVINO ----------
core = Core()
compiled_model = core.compile_model(core.read_model(model_path), "CPU")
input_layer, output_layer = compiled_model.input(0), compiled_model.output(0)

# ---------- Video ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头")

class_names = ["B1", "B2", "B3", "B4", "B5", "B7", "R1", "R2", "R3", "R4", "R5", "R7"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h0, w0 = frame.shape[:2]

    t0 = time.time()

    # ---------- Ultralytics 原版 letterbox ----------
    r = min(IMG_SIZE / h0, IMG_SIZE / w0)           # 缩放比例
    new_w, new_h = int(w0 * r), int(h0 * r)
    dw, dh = (IMG_SIZE - new_w) / 2, (IMG_SIZE - new_h) / 2  # 四周补边（float）
    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[int(dh):int(dh + new_h), int(dw):int(dw + new_w)] = img

    inp = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    # ---------- Inference ----------
    preds = compiled_model([inp])[output_layer].squeeze(0)   # (N,6)

    annotator = Annotator(frame)

    for *xyxy, conf, cls in preds:
        if conf < 0.25:
            continue
        x1, y1, x2, y2 = xyxy

        # ---------- 坐标反变换 ----------
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r
        # 裁到原图尺寸
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)

        label = f"{class_names[int(cls)]} {conf:.2f}"
        annotator.box_label([int(x1), int(y1), int(x2), int(y2)], label, color=(0, 255, 0))

    fps = 1.0 / (time.time() - t0 + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
    cv2.imshow("YOLOv8 + OpenVINO", annotator.result())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
