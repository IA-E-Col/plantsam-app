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
from typing import List, Tuple
import json

app = FastAPI()

device = "cuda"

if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="sam2/sam2_configs", version_base=None)

predictor = SAM2ImagePredictor(
    build_sam2("sam2_hiera_l.yaml", "models/sam2_hiera_large.pt", device=device)
)
predictor.model.load_state_dict(torch.load("models/BBS2_1024_2_epoch5.torch"))
model_yolo_1024 = YOLOv10("models/trainedyolov10.pt")

union_masks_storage = {}
intersection_masks_storage = {}
base_masks_storage = {}
initial_masks_storage = {}
final_masks_storage = {}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    try:
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

        # Vérifier que l'image a bien été générée
        if result_bgr is None:
            print("❌ Erreur: l'image traitée n'a pas été générée")
            # Retourner une image de fallback
            result_bgr = np.zeros_like(image_bgr)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)

        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        _, initial_mask = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)
        initial_masks_storage[file.filename] = initial_mask

        print(f"✅ Segmentation complète réussie, image renvoyée: {buf.getbuffer().nbytes} bytes")

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"❌ Erreur dans process_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        # Retourner une image noire en cas d'erreur
        blank_image = Image.new('RGB', (100, 100), color='black')
        buf = io.BytesIO()
        blank_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

def segment_with_multiple_points(image_bytes: bytes, positive_points: List[Tuple[int, int]], negative_points: List[Tuple[int, int]], start_type: str = "scratch"):
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    all_point_coords = []
    all_point_labels = []

    for point in positive_points:
        all_point_coords.append([point[0], point[1]])
        all_point_labels.append(1)

    for point in negative_points:
        all_point_coords.append([point[0], point[1]])
        all_point_labels.append(0)

    if not all_point_coords and not negative_points:
        if start_type == "scratch":
            segmented = np.zeros_like(image_bgr)
        else:
            segmented = image_bgr
    else:
        point_coords = np.array(all_point_coords)
        point_labels = np.array(all_point_labels)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False
                )
                mask = masks[np.argmax(scores)].astype(np.uint8)

        if start_type == "segmented" and len(positive_points) > 0:
            final_mask = mask
        else:
            final_mask = mask

        mask_rgb = np.stack((final_mask,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)

    return buf

@app.post("/segment_with_points")
async def segment_with_points(
        file: UploadFile = File(...),
        positive_points: str = Form("[]"),
        negative_points: str = Form("[]"),
        start_type: str = Form("scratch")
):
    image_bytes = await file.read()

    try:
        pos_points = json.loads(positive_points)
        neg_points = json.loads(negative_points)
    except json.JSONDecodeError:
        pos_points = []
        neg_points = []

    print(f"Points positifs: {pos_points}")
    print(f"Points négatifs: {neg_points}")
    print(f"Start type: {start_type}")

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_key = file.filename

    # Toujours préférer le masque final s'il existe (base la plus à jour)
    current_mask = None
    if image_key in final_masks_storage:
        current_mask = final_masks_storage[image_key]
        print("✅ Utilisation du masque final comme base pour les points")
    elif start_type == "segmented" and image_key in initial_masks_storage:
        current_mask = initial_masks_storage[image_key]
        print("Utilisation du masque initial comme base pour les points")

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    all_point_coords = []
    all_point_labels = []

    for point in pos_points:
        all_point_coords.append([point[0], point[1]])
        all_point_labels.append(1)

    for point in neg_points:
        all_point_coords.append([point[0], point[1]])
        all_point_labels.append(0)

    if not all_point_coords and not neg_points:
        if start_type == "scratch":
            segmented = np.zeros_like(image_bgr)
        else:
            if current_mask is not None:
                mask_rgb = np.stack((current_mask,) * 3, axis=-1)
                segmented = np.where(mask_rgb != 0, image_bgr, 0)
            else:
                segmented = image_bgr
    else:
        point_coords = np.array(all_point_coords)
        point_labels = np.array(all_point_labels)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False
                )
                new_mask = masks[np.argmax(scores)].astype(np.uint8)

        # Combine with current mask considering rectangle-removed areas
        if current_mask is not None:
            if len(pos_points) > 0:
                # For positive points, combine with current mask but preserve zeros (rectangle-removed areas)
                # This means new segments cannot appear in rectangle-removed areas
                combined_mask = np.logical_or(current_mask, new_mask).astype(np.uint8)
                final_mask = np.where(current_mask == 0, 0, combined_mask).astype(np.uint8)
            else:
                # For negative points, simply remove the area from current mask
                final_mask = np.logical_and(current_mask, np.logical_not(new_mask)).astype(np.uint8)
        else:
            final_mask = new_mask

        # Store the result for future operations
        final_masks_storage[image_key] = final_mask

        mask_rgb = np.stack((final_mask,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/clear_points")
async def clear_points(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_key = file.filename

    if image_key in union_masks_storage:
        del union_masks_storage[image_key]
    if image_key in intersection_masks_storage:
        del intersection_masks_storage[image_key]
    if image_key in base_masks_storage:
        del base_masks_storage[image_key]

    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    segmented = np.zeros_like(image_bgr)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/segment_union")
async def segment_union(
        file: UploadFile = File(...),
        x: int = Form(...),
        y: int = Form(...),
        point_count: int = Form(...),
        start_type: str = Form("scratch")
):
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        image_key = file.filename

        print(f"Segment union - Point: ({x}, {y}), Count: {point_count}, Start type: {start_type}")

        # Toujours préférer le masque final comme base s'il existe
        if point_count == 1 and image_key in final_masks_storage:
            base_mask = final_masks_storage[image_key]
            union_masks_storage[image_key] = base_mask.copy()
            print("✅ Utilisation du masque final comme base pour l'union")
        elif point_count == 1 and start_type == "segmented" and image_key in initial_masks_storage:
            initial_mask = initial_masks_storage[image_key]
            union_masks_storage[image_key] = initial_mask.copy()
            print("Utilisation du masque initial comme base pour l'union")

        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        point_coords = np.array([[x, y]])
        point_labels = np.array([1])

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True
                )
                best_mask_index = np.argmax(scores)
                current_mask = masks[best_mask_index].astype(np.uint8)

        if point_count == 1:
            if image_key in union_masks_storage:
                previous_mask = union_masks_storage[image_key]
                union_mask = np.logical_or(previous_mask, current_mask).astype(np.uint8)
                union_masks_storage[image_key] = union_mask
            else:
                union_masks_storage[image_key] = current_mask
        else:
            if image_key in union_masks_storage:
                previous_mask = union_masks_storage[image_key]
                union_mask = np.logical_or(previous_mask, current_mask).astype(np.uint8)
                union_masks_storage[image_key] = union_mask
            else:
                union_masks_storage[image_key] = current_mask

        final_mask = union_masks_storage[image_key]

        # TOUJOURS mettre à jour le masque final
        final_masks_storage[image_key] = final_mask

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        mask_rgb = np.stack((final_mask,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)

        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        segmented_pil = Image.fromarray(segmented_rgb)
        buf = io.BytesIO()
        segmented_pil.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"Erreur dans segment_union: {str(e)}")
        import traceback
        traceback.print_exc()
        return await return_error_image()

@app.post("/segment_intersection")
async def segment_intersection(
        file: UploadFile = File(...),
        x: int = Form(...),
        y: int = Form(...),
        point_count: int = Form(...),
        start_type: str = Form("scratch")
):
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_key = file.filename

    print(f"Segment intersection - Point: ({x}, {y}), Count: {point_count}, Start type: {start_type}")

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    point_coords = np.array([[x, y]])
    point_labels = np.array([0])

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True
            )
            best_mask_index = np.argmax(scores)
            negative_mask = masks[best_mask_index].astype(np.uint8)

    if point_count == 1 and image_key in final_masks_storage:
        base_mask = final_masks_storage[image_key]
        base_masks_storage[image_key] = base_mask
        print("✅ Utilisation du masque final comme base pour l'intersection")
    elif point_count == 1 and start_type == "segmented" and image_key in initial_masks_storage:
        base_mask = initial_masks_storage[image_key]
        base_masks_storage[image_key] = base_mask
        print("Utilisation du masque initial comme base pour l'intersection")
    elif point_count == 1 and image_key not in base_masks_storage:
        height, width = image_bgr.shape[:2]
        center_x, center_y = width // 2, height // 2

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_masks, base_scores, _ = predictor.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1]),
                    box=None,
                    multimask_output=True
                )
                base_mask = base_masks[np.argmax(base_scores)].astype(np.uint8)
        base_masks_storage[image_key] = base_mask

    if point_count == 1:
        if image_key in base_masks_storage:
            base_mask = base_masks_storage[image_key]
            intersection_mask = np.logical_and(base_mask, np.logical_not(negative_mask)).astype(np.uint8)
            intersection_masks_storage[image_key] = intersection_mask
            print(f"Premier point négatif - Masque de base modifié")
        else:
            height, width = image_bgr.shape[:2]
            center_x, center_y = width // 2, height // 2

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base_masks, base_scores, _ = predictor.predict(
                        point_coords=np.array([[center_x, center_y]]),
                        point_labels=np.array([1]),
                        box=None,
                        multimask_output=True
                    )
                    base_mask = base_masks[np.argmax(base_scores)].astype(np.uint8)
            base_masks_storage[image_key] = base_mask
            intersection_mask = np.logical_and(base_mask, np.logical_not(negative_mask)).astype(np.uint8)
            intersection_masks_storage[image_key] = intersection_mask
    else:
        if image_key in intersection_masks_storage:
            previous_mask = intersection_masks_storage[image_key]

            intersection_mask = np.logical_and(previous_mask, np.logical_not(negative_mask)).astype(np.uint8)
            intersection_masks_storage[image_key] = intersection_mask
            print(f"Point négatif supplémentaire - Masque précédent modifié")
        else:
            if image_key not in base_masks_storage:
                height, width = image_bgr.shape[:2]
                center_x, center_y = width // 2, height // 2

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        base_masks, base_scores, _ = predictor.predict(
                            point_coords=np.array([[center_x, center_y]]),
                            point_labels=np.array([1]),
                            box=None,
                            multimask_output=True
                        )
                        base_mask = base_masks[np.argmax(base_scores)].astype(np.uint8)
                base_masks_storage[image_key] = base_mask

            base_mask = base_masks_storage[image_key]
            intersection_mask = np.logical_and(base_mask, np.logical_not(negative_mask)).astype(np.uint8)
            intersection_masks_storage[image_key] = intersection_mask
            print(f"Masque d'intersection recréé - Point négatif appliqué")

    final_mask = intersection_masks_storage[image_key]

    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    mask_rgb = np.stack((final_mask,) * 3, axis=-1)
    segmented = np.where(mask_rgb != 0, image_bgr, 0)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/remove_rectangle")
async def remove_rectangle(
        file: UploadFile = File(...),
        x: int = Form(...),
        y: int = Form(...),
        width: int = Form(...),
        height: int = Form(...),
        start_type: str = Form("segmented")
):
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_key = file.filename

        print(f"=== REMOVE RECTANGLE DEBUG ===")
        print(f"Received - Coords: ({x}, {y}), Size: {width}x{height}, Start type: {start_type}")
        print(f"Image dimensions: {image_bgr.shape}")

        img_height, img_width = image_bgr.shape[:2]

        print(f"Image size: {img_width}x{img_height}")
        print(f"Rectangle covers: x={x}-{x+width} ({width}px), y={y}-{y+height} ({height}px)")

        x1 = max(0, min(x, img_width - 1))
        y1 = max(0, min(y, img_height - 1))
        x2 = max(0, min(x + width, img_width))
        y2 = max(0, min(y + height, img_height))

        actual_width = x2 - x1
        actual_height = y2 - y1

        print(f"Adjusted coords: ({x1}, {y1}), Size: {actual_width}x{actual_height}")

        if actual_width <= 0 or actual_height <= 0:
            print("❌ Rectangle invalide après ajustement")
            return await return_original_image(image_bgr)

        # Get the current mask states
        if image_key in final_masks_storage:
            current_mask = final_masks_storage[image_key].copy()
            print("✅ Using current mask state from storage")
        elif image_key in initial_masks_storage:
            current_mask = initial_masks_storage[image_key].copy()
            print("✅ Using initial mask as base")
        else:
            current_mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
            print("⚠️ No mask found, using full white mask")

        print(f"Mask shape: {current_mask.shape}, dtype: {current_mask.dtype}")
        
        # Force remove the rectangle area from the mask
        # This will override any previous segmentation in this area
        current_mask[y1:y2, x1:x2] = 0
        print(f"✅ Forced rectangle removal from mask")
        
        # Store rectangle removal result as final mask for future operations
        final_masks_storage[image_key] = current_mask

        mask_rgb = np.stack((current_mask,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)

        print(f"✅ Final image prepared")
        print(f"=======================")

        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        segmented_pil = Image.fromarray(segmented_rgb)
        buf = io.BytesIO()
        segmented_pil.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"❌ Erreur dans remove_rectangle: {str(e)}")
        import traceback
        traceback.print_exc()
        return await return_error_image()

async def return_original_image(image_bgr):
    segmented_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_rgb)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

async def return_error_image():
    blank_image = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    blank_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/clear_rectangles")
async def clear_rectangles(file: UploadFile = File(...)):
    try:
        image_key = file.filename

        if image_key in final_masks_storage:
            del final_masks_storage[image_key]

        # Si vous voulez revenir au masque initial
        if image_key in initial_masks_storage:
            final_masks_storage[image_key] = initial_masks_storage[image_key].copy()

        return {"status": "success", "message": "rectangles cleared"}
    except Exception as e:
        print(f"❌ Error clearing rectangles: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/apply_union")
async def apply_union(file: UploadFile = File(...), previous_mask: UploadFile = File(...)):
    try:
        # Read the current image
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_key = file.filename

        # Read and process the previous mask
        prev_mask_bytes = await previous_mask.read()
        prev_mask_pil = Image.open(io.BytesIO(prev_mask_bytes)).convert("RGB")
        prev_mask_bgr = cv2.cvtColor(np.array(prev_mask_pil), cv2.COLOR_BGR2GRAY)
        _, prev_mask_binary = cv2.threshold(prev_mask_bgr, 1, 1, cv2.THRESH_BINARY)

        # Get the current final mask
        if image_key not in final_masks_storage:
            return {"status": "error", "message": "No final mask found for the image"}

        current_mask = final_masks_storage[image_key]

        # Apply union operation
        union_result = np.logical_or(current_mask, prev_mask_binary).astype(np.uint8)
        final_masks_storage[image_key] = union_result

        # Create visualization
        mask_rgb = np.stack((union_result,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)
        
        # Convert to RGB and prepare response
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        segmented_pil = Image.fromarray(segmented_rgb)
        buf = io.BytesIO()
        segmented_pil.save(buf, format="PNG")
        buf.seek(0)

        print(f"✅ Union operation completed successfully")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"❌ Error in apply_union: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}