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

def segment_with_points(image_bytes: bytes, positive_points: List[Tuple[int, int]], negative_points: List[Tuple[int, int]], start_type: str = "scratch"):
    print("\n=== SEGMENT WITH POINTS DEBUG ===")
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    print(f"Points info:")
    print(f"- Positive points: {positive_points}")
    print(f"- Negative points: {negative_points}")
    print(f"- Start type: {start_type}")

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

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_key = file.filename

    print("\n=== SEGMENT WITH POINTS DEBUG ===")
    print(f"Positive points: {pos_points}")
    print(f"Negative points: {neg_points}")
    print(f"Start type: {start_type}")
    
    # Set up initial state
    print("\nChecking mask storage state:")
    print(f"- final_masks_storage has key: {image_key in final_masks_storage}")
    print(f"- base_masks_storage has key: {image_key in base_masks_storage}")
    print(f"- union_masks_storage has key: {image_key in union_masks_storage}")
    print(f"- initial_masks_storage has key: {image_key in initial_masks_storage}")

    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Get the base mask to use - prioritize base_masks_storage since it contains union results
    if image_key in base_masks_storage:
        print("✅ Using base mask from storage (contains union result)")
        base_mask = base_masks_storage[image_key].copy()
        print(f"Base mask values: {np.unique(base_mask)}")
    elif image_key in final_masks_storage:
        print("✅ Using final mask as base")
        base_mask = final_masks_storage[image_key].copy()
        print(f"Base mask values: {np.unique(base_mask)}")
    elif image_key in initial_masks_storage and start_type == "segmented":
        print("✅ Using initial mask as base")
        base_mask = initial_masks_storage[image_key].copy()
        print(f"Base mask values: {np.unique(base_mask)}")
    else:
        print("✅ Starting with empty mask")
        base_mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)

    print(f"Base mask values: {np.unique(base_mask)}")
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
            final_masks_storage[image_key] = np.zeros_like(base_mask)
        else:
            # Check if we have a base mask from a previous union operation
            if image_key in base_masks_storage:
                base_mask = base_masks_storage[image_key]
                print("✅ Using union result as base mask")
            
            mask_rgb = np.stack((base_mask,) * 3, axis=-1)
            segmented = np.where(mask_rgb != 0, image_bgr, 0)
            final_masks_storage[image_key] = base_mask.copy()
    else:
        point_coords = np.array(all_point_coords)
        point_labels = np.array(all_point_labels)
        
        # Check if we have a base mask from union operation
        if image_key in base_masks_storage:
            base_mask = base_masks_storage[image_key]
            print(f"✅ Using union result as base mask (values: {np.unique(base_mask)})")

        print(f"Predicting with points: {point_coords.tolist()}")
        if current_mask is not None:
            print(f"Using existing mask with values: {np.unique(current_mask)}")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=False
                )
                new_mask = masks[np.argmax(scores)].astype(np.uint8)

                print(f"Base mask before combining - unique values: {np.unique(base_mask)}")
                print(f"New mask from prediction - unique values: {np.unique(new_mask)}")

                if len(pos_points) > 0:
                    # For positive points, add new regions to the base mask
                    final_mask = np.logical_or(base_mask > 0, new_mask > 0).astype(np.uint8)
                    print("✅ Added new positive regions to base mask")
                else:
                    # For negative points, remove regions from the base mask
                    final_mask = np.logical_and(base_mask > 0, new_mask == 0).astype(np.uint8)
                    print("✅ Removed negative regions from base mask")

                print(f"Final combined mask - unique values: {np.unique(final_mask)}")

                # Update both storages with the new combined mask
                final_masks_storage[image_key] = final_mask.copy()
                base_masks_storage[image_key] = final_mask.copy()  # Important: keep base mask updated for future operations

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
    print("\n=== CLEAR POINTS ===")
    image_bytes = await file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_key = file.filename

    print(f"Storage state before clearing:")
    print(f"- final_masks_storage has key: {image_key in final_masks_storage}")
    print(f"- base_masks_storage has key: {image_key in base_masks_storage}")
    print(f"- union_masks_storage has key: {image_key in union_masks_storage}")

    # Store the existing union result if it exists
    existing_union = None
    if image_key in base_masks_storage:
        print("✅ Preserving existing union result")
        existing_union = base_masks_storage[image_key].copy()

    # Clear temporary storages
    if image_key in intersection_masks_storage:
        del intersection_masks_storage[image_key]

    # Restore the appropriate base mask
    if existing_union is not None:
        print("✅ Restoring previous union result as base")
        final_masks_storage[image_key] = existing_union.copy()
        base_masks_storage[image_key] = existing_union.copy()
        print(f"Restored mask values: {np.unique(existing_union)}")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        mask_rgb = np.stack((existing_union,) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)
    elif image_key in initial_masks_storage:
        print("✅ Restoring initial mask as base")
        final_masks_storage[image_key] = initial_masks_storage[image_key].copy()
        base_masks_storage[image_key] = initial_masks_storage[image_key].copy()
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        mask_rgb = np.stack((initial_masks_storage[image_key],) * 3, axis=-1)
        segmented = np.where(mask_rgb != 0, image_bgr, 0)
    else:
        print("✅ Resetting to empty mask")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        segmented = np.zeros_like(image_bgr)

    print("✅ Points cleared successfully")
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

        print(f"\n=== SEGMENT UNION DEBUG ===")
        print(f"Point: ({x}, {y}), Count: {point_count}, Start type: {start_type}")
        print(f"Storage state:")
        print(f"- final_masks_storage has key: {image_key in final_masks_storage}")
        print(f"- base_masks_storage has key: {image_key in base_masks_storage}")
        print(f"- union_masks_storage has key: {image_key in union_masks_storage}")
        if image_key in final_masks_storage:
            print(f"- final_mask values: {np.unique(final_masks_storage[image_key])}")
        if image_key in base_masks_storage:
            print(f"- base_mask values: {np.unique(base_masks_storage[image_key])}")

        # Toujours préférer le masque final comme base s'il existe
        if point_count == 1 and image_key in final_masks_storage:
            base_mask = final_masks_storage[image_key]
            union_masks_storage[image_key] = base_mask.copy()
            print("✅ Using final mask as base for union operation")
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
        print("\n=== APPLY UNION DEBUG ===")
        
        # Read the current image
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_key = file.filename
        
        print(f"Initial storage state:")
        print(f"- final_masks_storage has key: {image_key in final_masks_storage}")
        print(f"- base_masks_storage has key: {image_key in base_masks_storage}")
        print(f"- union_masks_storage has key: {image_key in union_masks_storage}")
        if image_key in final_masks_storage:
            print(f"- final_mask values: {np.unique(final_masks_storage[image_key])}")
        if image_key in base_masks_storage:
            print(f"- base_mask values: {np.unique(base_masks_storage[image_key])}")

        # Read and process the previous mask to binary (0 or 1)
        prev_mask_bytes = await previous_mask.read()
        prev_mask_pil = Image.open(io.BytesIO(prev_mask_bytes)).convert("RGB")
        prev_mask_bgr = cv2.cvtColor(np.array(prev_mask_pil), cv2.COLOR_BGR2GRAY)
        _, prev_mask_binary = cv2.threshold(prev_mask_bgr, 1, 1, cv2.THRESH_BINARY)

        # Get the current final mask
        if image_key not in final_masks_storage:
            current_mask = np.zeros_like(prev_mask_binary)
        else:
            current_mask = final_masks_storage[image_key].copy()

        # Make sure both masks are strictly binary (0 or 1)
        current_mask = (current_mask > 0).astype(np.uint8)
        prev_mask_binary = (prev_mask_binary > 0).astype(np.uint8)

        # Apply union operation
        union_result = np.logical_or(current_mask, prev_mask_binary).astype(np.uint8)
        print(f"\nUnion operation complete - values in result: {np.unique(union_result)}")
        
        # Store the result in all storages to maintain consistent state
        final_masks_storage[image_key] = union_result.copy()
        base_masks_storage[image_key] = union_result.copy()  # This is crucial - it becomes the base for future operations
        union_masks_storage[image_key] = union_result.copy()

        # Clear any temporary state to ensure clean operations
        if image_key in intersection_masks_storage:
            del intersection_masks_storage[image_key]

        print("\nMask storage state updated:")
        print(f"- final_masks_storage contains mask with values: {np.unique(final_masks_storage[image_key])}")
        print(f"- base_masks_storage contains mask with values: {np.unique(base_masks_storage[image_key])}")
        print(f"- union_masks_storage contains mask with values: {np.unique(union_masks_storage[image_key])}")
        print("✅ Union result stored and ready for future operations")

        print(f"Union result unique values: {np.unique(union_result)}")
        print("✅ Union operation completed and stored in final_masks_storage")

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