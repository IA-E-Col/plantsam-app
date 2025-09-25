import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from segment_anything import sam_model_registry
import torch
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from statistics import mean
from utils import *
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
import monai
x
def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


device = "cuda"


checkpoint = "path/to/checkpoint"
model_cfg = "sam2_hiera_l.yaml"
sam_model = build_sam2(model_cfg, checkpoint, device=device)


sam_model = torch.nn.DataParallel(sam_model) 
sam_model.to(device)  

predictor = SAM2ImagePredictor(sam_model.module)

predictor.model.sam_mask_decoder.train(True) 
predictor.model.sam_prompt_encoder.train(True)

size = 1024
img_patch_size = (size, size, 3)
step = size

GT_PATH = "path/to/ground_truth_masks"
IMAGES_PATH = "path/to/images"

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler() 

mask_params = list(sam_model.module.sam_mask_decoder.parameters())
print(f"Mask Decoder Parameters: {len(mask_params)}")

step_size = 30
gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

kernel_size = 5
kernel = np.ones((kernel_size,kernel_size), np.uint8)
num_epochs = 20
losses = []
val_losses = []
visualization_data = []

output_dir = "views"
os.makedirs(output_dir, exist_ok=True)

# Loading datas
print("loading prompts")

fichier = 'very_cleaned_dataset_1476.json'
train_data = read_json(fichier)

print("prompts loaded")

# Split data into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(train_data))), test_size=0.2, random_state=42)

for epoch in range(num_epochs):
    epoch_losses = []
    epoch_val_losses = []
    image_name = ""

    sam_model.train()
    for k in tqdm(train_indices):
        if image_name != train_data[k]:
            try:
                current_image = cv2.imread(IMAGES_PATH + "/" + train_data[k])
                if current_image is None:
                    print(f"Error: Image {IMAGES_PATH + '/' + train_data[k]} could not be loaded.")
                    continue
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                gt_grayscale = cv2.imread(GT_PATH + "/" + train_data[k], cv2.IMREAD_GRAYSCALE)

                if gt_grayscale is None:
                    print(f"Error: Ground truth mask {GT_PATH + '/' + train_data[k]} could not be loaded.")
                    continue

            except cv2.error as e:
                print(f"Error reading image {IMAGES_PATH + '/' + train_data[k]}: {e}")
                continue
        
        image_patches = patchify_with_border_handling(current_image, img_patch_size, step=step)
        gt_mask_patches = patchify_with_border_handling(gt_grayscale, img_patch_size[:2], step=step)

        num_patches_y, num_patches_x = image_patches.shape[:2]

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                current_patch = image_patches[i, j, 0]
                current_gt_mask = gt_mask_patches[i, j]

                percentage_zeros = np.mean(current_gt_mask == 0) * 100
                if percentage_zeros > 99:
                    continue

                ###### Erosion + dilatation

                
                eroded_mask = cv2.erode(current_gt_mask, kernel, iterations=1)
                current_gt_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

                ########


                bbox = get_bounding_box(current_gt_mask)


                with torch.cuda.amp.autocast():
                    predictor.set_image(current_patch)
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                        point_coords=None, 
                        point_labels=None, 
                        box=bbox,
                        mask_logits=None, 
                        normalize_coords=True
                    )
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=None, 
                        boxes=unnorm_box, 
                        masks=None
                    )

                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
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

    # Validation step
    sam_model.eval()
    with torch.no_grad():
        for k in tqdm(val_indices):
            if image_name != train_data[k]:
                try:
                    current_image = cv2.imread(IMAGES_PATH + "/" + train_data[k])
                    if current_image is None:
                        print(f"Error: Image {IMAGES_PATH + '/' + train_data[k]} could not be loaded.")
                        continue
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                    gt_grayscale = cv2.imread(GT_PATH + "/" + train_data[k], cv2.IMREAD_GRAYSCALE)
                    if gt_grayscale is None:
                        print(f"Error: Ground truth mask {GT_PATH + '/' + train_data[k]} could not be loaded.")
                        continue

                except cv2.error as e:
                    print(f"Error reading image {IMAGES_PATH + '/' + train_data[k]}: {e}")
                    continue
            
            image_patches = patchify_with_border_handling(current_image, img_patch_size, step=step)
            gt_mask_patches = patchify_with_border_handling(gt_grayscale, img_patch_size[:2], step=step)

            for i in range(image_patches.shape[0]):
                for j in range(image_patches.shape[1]):
                    current_patch = image_patches[i, j, 0]
                    current_gt_mask = gt_mask_patches[i, j]

                    percentage_zeros = np.mean(current_gt_mask == 0) * 100
                    if percentage_zeros > 99:
                        continue

                    eroded_mask = cv2.erode(current_gt_mask, kernel, iterations=1)
                    current_gt_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

                    ########


                    bbox = get_bounding_box(current_gt_mask)
                    
                    with torch.cuda.amp.autocast():
                        predictor.set_image(current_patch)
                        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                            point_coords=None, 
                            point_labels=None, 
                            box=bbox,
                            mask_logits=None, 
                            normalize_coords=True
                        )
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=None, 
                            boxes=unnorm_box, 
                            masks=None
                        )

                        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
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

    # Save model checkpoint
    chemin_fichier = f"BBS2_{size}_6_epoch{epoch}_SAMDS.torch"
    torch.save(predictor.model.state_dict(), chemin_fichier)

    print(f'EPOCH: {epoch}')
    print(f'MSE training loss: {mean(epoch_losses)}')
    print(f'MSE validation loss: {mean(epoch_val_losses)}')

    # Plot training loss
    mean_losses = [mean(x) for x in losses]
    plt.plot(list(range(len(mean_losses))), mean_losses, label='Training Loss')

    # Plot validation loss
    mean_val_losses = [mean(x) for x in val_losses]
    plt.plot(list(range(len(mean_val_losses))), mean_val_losses, label='Validation Loss')

    plt.title('Mean epoch loss (train & val)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./BBS2_{size}_loss_graph_train_val__SAMDS.jpg")
    plt.close()

print("Model save at :", chemin_fichier)
