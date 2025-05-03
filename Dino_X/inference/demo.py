# dds cloudapi for DINO-X
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

# using supervision for visualization
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from pycocotools import mask as mask_utils
from rle_util import rle_to_array
import json

"""
Hyper Parameters
"""
API_TOKEN = "API"
IMG_PATH = "/home/ubuntu/m15kh/own/proj/Generator/img_tst/tst2.png"
TEXT_PROMPT =  "person"

OUTPUT_DIR = Path("/home/ubuntu/m15kh/own/proj/Generator/experiments/DINO-X-API/out/tst2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Prompting DINO-X with Text for Box and Mask Generation with Cloud API
"""

# Step 1: initialize the config
token = API_TOKEN
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: Run V2 task
# if you are processing local image file, upload them to DDS server to get the image url
image_url = client.upload_file(IMG_PATH)

v2_task = V2Task(
    api_path="/v2/task/dinox/detection",
    api_body={
        "model": "DINO-X-1.0",  # 使用适当的模型名称
        "image": image_url,
        "prompt": {
            "type": "text",
            "text": TEXT_PROMPT
        },
        "targets": ["bbox", "mask"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
    }
)

client.run_task(v2_task)
result = v2_task.result

objects = result["objects"]

"""
Visualization
"""

# decode the prediction results

classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
class_name_to_id = {name: id for id, name in enumerate(classes)}
class_id_to_name = {id: name for name, id in class_name_to_id.items()}

boxes = []
masks = []
confidences = []
class_names = []
class_ids = []

for idx, obj in enumerate(objects):
    boxes.append(obj["bbox"])
    masks.append(rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1]).reshape(obj["mask"]["size"]))
    confidences.append(obj["score"])
    cls_name = obj["category"].lower().strip()
    class_names.append(cls_name)
    class_ids.append(class_name_to_id[cls_name])

boxes = np.array(boxes)
masks = np.array(masks)
class_ids = np.array(class_ids)
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy = boxes,
    mask = masks.astype(bool),
    class_id = class_ids,
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image.jpg"), annotated_frame)


mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_demo_image_with_mask.jpg"), annotated_frame)

print(f"Annotated image has already been saved to {OUTPUT_DIR}")

# Save bounding boxes and masks to a JSON file
output_data = []
for idx, obj in enumerate(objects):
    output_data.append({
        "bbox": obj["bbox"],
        "mask": obj["mask"],
        "score": obj["score"],
        "category": obj["category"]
    })

json_output_path = os.path.join(OUTPUT_DIR, "detections.json")
with open(json_output_path, "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Bounding boxes and masks have been saved to {json_output_path}")
