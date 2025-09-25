
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
import torch
from tqdm import tqdm
from statistics import mean
from utils import *
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import monai
import hydra

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def main(args):

    
        # --- Ajoute cette partie ---
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="sam2/sam2_configs", version_base=None)
    # ---------------------------


    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    sam_model = torch.nn.DataParallel(sam_model) 
    sam_model.to(device)  

    predictor = SAM2ImagePredictor(sam_model.module)
    predictor.model.sam_mask_decoder.train(True) 
    predictor.model.sam_prompt_encoder.train(True)

    size = args.patch_size
    img_patch_size = (size, size, 3)
    step = size

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() 

    step_size = 30
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    kernel_size = 5
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    num_epochs = args.epochs
    losses, val_losses = [], []

    os.makedirs(args.output_dir, exist_ok=True)

    train_data = read_json(args.dataset_json)
    train_indices, val_indices = train_test_split(list(range(len(train_data))), test_size=0.2, random_state=42)

    for epoch in range(num_epochs):
        epoch_losses, epoch_val_losses = [], []
        image_name = ""
        sam_model.train()

        for k in tqdm(train_indices):
            if image_name != train_data[k]:
                img_path = os.path.join(args.images_path, train_data[k])
                gt_path = os.path.join(args.gt_path, train_data[k])

                if not os.path.exists(img_path) or not os.path.exists(gt_path):
                    print(f"missing file : {img_path} ou {gt_path}")
                    continue

                try:
                    current_image = cv2.imread(img_path)
                    if current_image is None:
                        print(f"Impossible to read the image : {img_path}")
                        continue
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                    gt_grayscale = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_grayscale is None:
                        print(f"Impossible to read the mask : {gt_path}")
                        continue
                except cv2.error as e:
                    print(f"Error OpenCV : {e} for {img_path}")
                    continue


            image_patches = patchify_with_border_handling(current_image, img_patch_size, step=step)
            gt_mask_patches = patchify_with_border_handling(gt_grayscale, img_patch_size[:2], step=step)

            for i in range(image_patches.shape[0]):
                for j in range(image_patches.shape[1]):
                    current_patch = image_patches[i, j, 0]
                    current_gt_mask = gt_mask_patches[i, j]
                    if np.mean(current_gt_mask == 0) * 100 > 99:
                        continue
                    current_gt_mask = cv2.dilate(cv2.erode(current_gt_mask, kernel, 1), kernel, 1)
                    bbox = get_bounding_box(current_gt_mask)

                    with torch.cuda.amp.autocast():
                        predictor.set_image(current_patch)
                        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                            point_coords=None, point_labels=None, box=bbox,
                            mask_logits=None, normalize_coords=True
                        )
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=None, boxes=unnorm_box, masks=None
                        )
                        high_res_features = [f[-1].unsqueeze(0) for f in predictor._features["high_res_feats"]]
                        low_res_masks, *_ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            repeat_image=None,
                            multimask_output=False,
                            high_res_features=high_res_features,
                        )
                        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                    gt_mask_resized = torch.from_numpy(np.resize(current_gt_mask, (1,1,prd_masks.shape[-2], prd_masks.shape[-1]))).to(device)
                    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
                    loss = loss_fn(prd_masks, gt_binary_mask)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_losses.append(loss.item())
            image_name = train_data[k]

        scheduler.step()

        sam_model.eval()
        with torch.no_grad():
            for k in tqdm(val_indices):
                if image_name != train_data[k]:
                    try:
                        current_image = cv2.imread(os.path.join(args.images_path, train_data[k]))
                        if current_image is None:
                            continue
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                        gt_grayscale = cv2.imread(os.path.join(args.gt_path, train_data[k]), cv2.IMREAD_GRAYSCALE)
                        if gt_grayscale is None:
                            continue
                    except cv2.error:
                        continue

                image_patches = patchify_with_border_handling(current_image, img_patch_size, step=step)
                gt_mask_patches = patchify_with_border_handling(gt_grayscale, img_patch_size[:2], step=step)

                for i in range(image_patches.shape[0]):
                    for j in range(image_patches.shape[1]):
                        current_patch = image_patches[i, j, 0]
                        current_gt_mask = gt_mask_patches[i, j]
                        if np.mean(current_gt_mask == 0) * 100 > 99:
                            continue
                        current_gt_mask = cv2.dilate(cv2.erode(current_gt_mask, kernel, 1), kernel, 1)
                        bbox = get_bounding_box(current_gt_mask)

                        with torch.cuda.amp.autocast():
                            predictor.set_image(current_patch)
                            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                                point_coords=None, point_labels=None, box=bbox,
                                mask_logits=None, normalize_coords=True
                            )
                            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                                points=None, boxes=unnorm_box, masks=None
                            )
                            high_res_features = [f[-1].unsqueeze(0) for f in predictor._features["high_res_feats"]]
                            low_res_masks, *_ = predictor.model.sam_mask_decoder(
                                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                repeat_image=None,
                                multimask_output=False,
                                high_res_features=high_res_features,
                            )
                            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                        gt_mask_resized = torch.from_numpy(np.resize(current_gt_mask, (1,1,prd_masks.shape[-2], prd_masks.shape[-1]))).to(device)
                        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
                        val_loss = loss_fn(prd_masks, gt_binary_mask)
                        epoch_val_losses.append(val_loss.item())

        losses.append(epoch_losses)
        val_losses.append(epoch_val_losses)

        model_file = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
        torch.save(predictor.model.state_dict(), model_file)

        print(f"EPOCH {epoch} — Train Loss: {mean(epoch_losses):.4f} — Val Loss: {mean(epoch_val_losses):.4f}")

        plt.plot([mean(x) for x in losses], label='Train Loss')
        plt.plot([mean(x) for x in val_losses], label='Val Loss')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"loss_plot_epoch_{epoch}.png"))
        plt.close()

    print("Training complete. Last model saved at:", model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM2 with erosion/dilation preprocessing and patching")
    parser.add_argument("--images_path", type=str, required=True, help="Path to input RGB images")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to .json file with training filenames")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to SAM2 model configuration YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 model checkpoint file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save models and plots")
    parser.add_argument("--patch_size", type=int, default=1024, help="Patch size to split images and masks")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    main(args)
