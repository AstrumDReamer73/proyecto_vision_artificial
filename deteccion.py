from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")  

start_time = time.time()

model.train(
    data="data.yaml",
    epochs=100,         
    imgsz=640,
    batch=16,
    project="yolo",
    name="experimento1"
)

end_time = time.time()
print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")

val_results = model.val(
    data="data.yaml",
    model="yolo/experimento1/weights/best.pt"
)

print(f"mAP@0.5: {val_results.metrics['map50']:.3f}")
print(f"Precisión: {val_results.metrics['precision']:.3f}")
print(f"Recall: {val_results.metrics['recall']:.3f}")

test_results = model.val(
    data="data.yaml",
    split="test",
    model="yolo/experimento1/weights/best.pt"
)