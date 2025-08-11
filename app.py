import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import visualize_object_predictions
from pycocotools.coco import COCO
from PIL import Image
from numpy import asarray
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import pandas as pd
import json

st.title("Sistem Deteksi Wajah SAHI")

def coco_to_xyxy(coco_box):
    x, y, width, height = coco_box
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return [x1, y1, x2, y2]

def yolo_prediction(yolo_model, input_img):
    model = YOLO(yolo_model)
    results = model.predict(source=input_img, imgsz=640, show_labels=False, show_conf=False, augment=False)

    return results
    
def sahi_prediction(image, slice_width, slice_height, ovw_ratio, ovh_ratio):
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=ovh_ratio,
        overlap_width_ratio=ovw_ratio,
        postprocess_class_agnostic=True,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=2,
    )
    return result

def calculate_iou(pred_bbox, gt_bbox):
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    # Intersection
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    # Intersection Area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Area masing-masing bounding box
    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)

    # Union Area
    union_area = pred_area + gt_area - inter_area

    # IoU
    return inter_area / union_area if union_area != 0 else 0

def match_bb(pred_bb, gt_bb, iou_threshold):
    predicted_boxes = pred_bb
    groundtruth_boxes = gt_bb
    
    iou_matrix = np.zeros((len(predicted_boxes), len(groundtruth_boxes)))
    for x, pbb in enumerate(predicted_boxes):
        for y, gtbb in enumerate(groundtruth_boxes):
            iou_matrix[x, y] = calculate_iou(pbb, gtbb)
    
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    
    matched_pairs = []
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matched_pairs.append((predicted_boxes[row], groundtruth_boxes[col], iou_matrix[row, col]))
    
    return matched_pairs

model_path = "yolov850.pt"
model_path1 = "yolov8n_100e.pt"
model_path2 = "yolov8l_100e.pt"
model_path3 = YOLO("yolov8l_100e.pt")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path2,
    confidence_threshold=0.5,
    device="cuda:0",
)

coco_data = None
with open('data uji sahi 640/test/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

ground_truth_annotations = coco_data['annotations']
ground_truth_boxes = []
for ann in ground_truth_annotations:
    bbox = ann['bbox']
    ground_truth_boxes.append(coco_to_xyxy(bbox))

progress_bar = st.progress(0)
status_text = st.empty()
num = 0
iou_threshold = 0.1
table_col_name = ['Pred BBoxes', 'GT BBoxes', 'IoU']

uploaded_files = st.file_uploader("Pilih file gambar", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    msg_text = st.empty()
    msg_text.text("Deteksi Dimulai")
    num_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        image = Image.open(file)

        image_np = asarray(image.convert('RGB'))

        start_time_yolo = time.time()
        yolo_results = yolo_prediction(model_path2, image_np)
        yolo_time = time.time() - start_time_yolo

        yolo_visual = yolo_results[0].plot()
        plt.axis('off')
        
        st.image(yolo_visual, caption=f"{file.name} (Deteksi: {yolo_time:.2f} detik)", use_column_width=True)
        st.write(f"Waktu deteksi YOLO untuk {file.name}: {yolo_time:.2f} detik")

        yolo_pred_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            
        yolo_matched_bb = match_bb(yolo_pred_boxes, ground_truth_boxes, iou_threshold)

        yolo_iou_val=[item[2] for item in yolo_matched_bb]
        yolo_avg_iou=np.mean(yolo_iou_val)
        
        yolo_df = pd.DataFrame(yolo_matched_bb, columns=table_col_name)
        st.table(yolo_df)
        
        st.write(f"Total Predicted Bounding Box: {len(yolo_pred_boxes)}, Total Matched Bounding Box: {len(yolo_matched_bb)} ===")
        st.write(f"Rata-rata IoU = {yolo_avg_iou:.2f}")
        
        start_time = time.time()
        predicted_img = sahi_prediction(image_np, 180, 180, 0.2, 0.2)
        detection_time = time.time() - start_time
        
        visualize_object_predictions(
            image_np,
            object_prediction_list = predicted_img.object_prediction_list,
            hide_labels = True,
            hide_conf = False,
            output_dir = 'export/',
            file_name = 'temp_image',
            export_format = 'jpg'
        )
    
        visualized_image = "export/temp_image.jpg"
        st.image(visualized_image, caption=f"{file.name} (Deteksi: {detection_time:.2f} detik)", use_column_width=True)
        st.write(f"Waktu deteksi untuk {file.name}: {detection_time:.2f} detik")
        
        st.write("Bounding Box Deteksi:")
     
        sahi_predicted_boxes = []
        for obj in predicted_img.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            sahi_predicted_boxes.append(bbox)

        sahi_matched_bb = match_bb(sahi_predicted_boxes, ground_truth_boxes, iou_threshold)

        sahi_iou_val=[item[2] for item in sahi_matched_bb]
        sahi_avg_iou=np.mean(sahi_iou_val)
        
        sahi_df = pd.DataFrame(sahi_matched_bb, columns=table_col_name)
        st.table(sahi_df)        
        
        st.write(f"Total of Predicted Bounding Box: {len(sahi_predicted_boxes)}, Total Matched Bounding Box: {len(sahi_matched_bb)} ===")
        st.write(f"Mean IoU = {sahi_avg_iou:.2f}")
        
        progress_bar.progress((i + 1) / num_files)
        status_text.text(f"Memproses gambar {i + 1} dari {num_files}")

        time.sleep(0.5)

    msg_text.text("Deteksi Selesai.")
    
else:
    st.write("Silakan unggah file gambar.")
