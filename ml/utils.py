from patchify import patchify
import numpy as np
import json
import scipy

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def patchify_with_border_handling(image, img_patch_size, step):
    if len(image.shape) == 2 :
        H, W = image.shape
        
        # Calculer le nombre de patches nécessaires
        num_patches_h = int((H) / step) + 1

        num_patches_w = int((W) / step) + 1
        
        # Ajouter du padding pour que les dimensions soient des multiples de step
        pad_h = (num_patches_h * step) - H
        pad_w = (num_patches_w * step) - W
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        
        # Patchify l'image paddée
        image_patches = patchify(image_padded, img_patch_size, step=step)
        
        return image_patches
    else:
        H, W, _ = image.shape
    
        # Calculer le nombre de patches nécessaires
        num_patches_h = int((H) / step) + 1

        num_patches_w = int((W) / step) + 1
        
        # Ajouter du padding pour que les dimensions soient des multiples de step
        pad_h = (num_patches_h * step) - H
        pad_w = (num_patches_w * step) - W
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        # Patchify l'image paddée
        image_patches = patchify(image_padded, img_patch_size, step=step)
        
        return image_patches
    



def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    

    return bbox

def get_bounding_boxes(ground_truth_map, min_width=20, min_height=20):
    labeled_map, num_features = scipy.ndimage.label(ground_truth_map)

    bounding_boxes = []
    
    for feature_num in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled_map == feature_num)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)


        width = x_max - x_min
        height = y_max - y_min

        if width < min_width or height < min_height:
            continue


        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        
        bbox = [x_min, y_min, x_max, y_max]
        bounding_boxes.append(bbox)

    return bounding_boxes