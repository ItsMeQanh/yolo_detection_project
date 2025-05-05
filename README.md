<img width="110" alt="image" src="https://github.com/user-attachments/assets/9af13fbe-2164-4dae-8f1b-3bd9cd0f1290" /># 🧠 YOLOv8 Object Detection with FiftyOne

This project runs object detection on the COCO-2017 validation dataset using YOLOv8n, visualizes predictions with FiftyOne, and evaluates the results using COCO-style metrics.

---

## ✅ Steps Performed

1. Loaded `coco-2017` validation dataset (50 images)
2. Ran inference with `yolov8n.pt` using Ultralytics' YOLOv8
3. Converted detections to FiftyOne format
4. Visualized bounding boxes with FiftyOne
5. Evaluated performance using `dataset.evaluate_detections()`:
   - **Accuracy**: 0.47  
   - **Precision**: 0.74  
   - **Recall**: 0.57  
   - **F1 Score**: 0.64  
6. Exported predictions to `output_dir/`
![Uploading image.png…]()

---

## 📁 Project Structure

```
yolo_detection_project/
├── run_yolo_predictions.py         # Main Python script  
├── yolov8n.pt                      # YOLOv8n model weights  
├── output_dir/                     # Exported predictions and metadata  
│   ├── data/  
│   ├── fields/  
│   ├── metadata.json  
│   └── samples.json  
├── screenshots/                    # Proof of inference + results (FiftyOne GUI, metrics)  
└── README.md                       # You're reading it  
```

---

---

