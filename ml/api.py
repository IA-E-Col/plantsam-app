from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import os
import cv2
import torch
import numpy as np
import io
from PIL import Image
from ultralytics import YOLOv10
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from predict_mask import process_image, patchify_with_border_handling, is_contained_within
import hydra
from omegaconf import OmegaConf

app = FastAPI()

device = "cuda"

if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="sam2/sam2_configs", version_base=None)


predictor = SAM2ImagePredictor(
    build_sam2("sam2_hiera_l.yaml", "models/sam2_hiera_large.pt", device=device)
)
predictor.model.load_state_dict(torch.load("models/BBS2_1024_2_epoch5.torch"))
model_yolo_1024 = YOLOv10("models/trainedyolov10.pt")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):

    image_bytes = await file.read()
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    size = 1024
    step = size
    img_patch_size = (size, size, 3)

    output_path = os.path.join("demo", file.filename)
    os.makedirs("demo", exist_ok=True)

    cv2.imwrite("tmp.png", image_bgr)
    process_image("tmp.png", output_path, predictor, model_yolo_1024, size, step, img_patch_size)

    result_bgr = cv2.imread(output_path)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

def segment_with_point_prompt(image_bytes: bytes, x: int, y: int, label: int):

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    point_coords = np.array([[x, y]])
    point_labels = np.array([label])  # 1 = objet, 0 = fond

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False
            )
            mask = masks[np.argmax(scores)].astype(np.uint8)

    mask_rgb = np.stack((mask,) * 3, axis=-1)
    segmented = np.where(mask_rgb != 0, image_bgr, 0)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)

    return buf


@app.post("/positive_point")
async def positive_point(
        file: UploadFile = File(...),
        x: int = Form(...),
        y: int = Form(...)
):
    image_bytes = await file.read()
    buf = segment_with_point_prompt(image_bytes, x, y, label=1)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/negative_point")
async def negative_point(
        file: UploadFile = File(...),
        x: int = Form(...),
        y: int = Form(...)
):
    image_bytes = await file.read()
    buf = segment_with_point_prompt(image_bytes, x, y, label=0)
    return StreamingResponse(buf, media_type="image/png")
