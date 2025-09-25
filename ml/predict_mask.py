import os
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from patchify import patchify
from ultralytics import YOLOv10
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from omegaconf import OmegaConf

def calculate_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        raise ValueError("Both masks must have the same shape")
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

def is_contained_within(box1, box2):
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

def patchify_with_border_handling(image, img_patch_size, step):
    H, W = image.shape[:2]
    num_patches_h = (H // step) + 1
    num_patches_w = (W // step) + 1
    pad_h = (num_patches_h * step) - H
    pad_w = (num_patches_w * step) - W
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    return patchify(image_padded, img_patch_size, step=step)

def process_image(image_path, output_path, predictor, model_yolo_1024, size, step, img_patch_size):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Cannot read image : {image_path}")
        return

    # Découpage en patchs
    image_patches = patchify_with_border_handling(image_bgr, img_patch_size, step=step)
    num_patches_y, num_patches_x = image_patches.shape[:2]
    patch_height, patch_width = image_patches.shape[3:5]
    reconstructed_mask_full = np.zeros((num_patches_y * patch_height, num_patches_x * patch_width), dtype=np.uint8)

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch_bgr = image_patches[i, j, 0]
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

            # Détection YOLO
            results = model_yolo_1024(source=patch_rgb, conf=0.25, save=False)[0].boxes.xyxyn.tolist()
            final_patch_mask = np.zeros((size, size), dtype=np.uint8)
            non_contained_boxes = []


            for box in results:
                box = [int(elem * size) for elem in box]
                if not any(is_contained_within(box, existing_box) for existing_box in non_contained_boxes):
                    non_contained_boxes.append(box)
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            predictor.set_image(patch_rgb)
                            masks, scores, _ = predictor.predict(
                                point_coords=None, point_labels=None,
                                box=np.array(box)[None, :], multimask_output=False)
                            prediction = masks[np.argmax(scores)].astype(np.uint8)
                    final_patch_mask = np.maximum(final_patch_mask, cv2.resize(prediction, (size, size)))

            # Reconstruction du masque global
            y_start, y_end = i * patch_height, (i + 1) * patch_height
            x_start, x_end = j * patch_width, (j + 1) * patch_width
            reconstructed_mask_full[y_start:y_end, x_start:x_end] = final_patch_mask

    # Recadrage à la taille originale
    H, W, _ = image_bgr.shape
    reconstructed_mask_full = reconstructed_mask_full[:H, :W]

    # Application du masque sur l'image d'origine
    reconstructed_mask_rgb = np.stack((reconstructed_mask_full,) * 3, axis=-1)  # [H, W, 3]
    output_image_bgr = np.where(reconstructed_mask_rgb != 0, image_bgr, 0)

    cv2.imwrite(output_path, output_image_bgr)
    print(f"Treated image : {os.path.basename(output_path)}")

def main():

        # --- Ajoute cette partie ---
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="sam2/sam2_configs", version_base=None)
    # ---------------------------

    parser = argparse.ArgumentParser(description="Segmentation of patches in images using SAM2 and YOLOv10")
    parser.add_argument("input_folder", help="Path to the input folder containing images")
    parser.add_argument("--output_folder", default="./demo", help="Path to the output folder for segmented images")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    device = "cuda"
    size = 1024
    step = size
    img_patch_size = (size, size, 3)
    checkpoint = "models/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    print("Loading models...")
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.load_state_dict(torch.load("models/BBS2_1024_2_epoch5.torch"))
    model_yolo_1024 = YOLOv10("models/trainedyolov10.pt")

    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not filenames:
        print("No images found in the input folder.")
        return

    for filename in filenames:
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(image_path, output_path, predictor, model_yolo_1024, size, step, img_patch_size)

if __name__ == "__main__":
    main()
