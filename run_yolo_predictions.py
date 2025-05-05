from ultralytics import YOLO
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.labels as fol

# Load dataset
dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=50)

# Ensure metadata is available
dataset.compute_metadata()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run predictions
results = model.predict([s.filepath for s in dataset], verbose=False)

# Convert predictions to FiftyOne format
for sample, result in zip(dataset, results):
    detections = []
    for box in result.boxes:
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        w = x2 - x1
        h = y2 - y1
        detections.append(
            fol.Detection(
                label=label,
                confidence=confidence,
                bounding_box=[x1 / sample.metadata.width,
                              y1 / sample.metadata.height,
                              w / sample.metadata.width,
                              h / sample.metadata.height]
            )
        )
    sample["predictions"] = fol.Detections(detections=detections)
    sample.save()

# Launch FiftyOne app
session = fo.launch_app(dataset)
