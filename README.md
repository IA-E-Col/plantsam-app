# PlantSAM-App
**Semi-Automatic Annotation Tool for Herbarium Image Segmentation**

## Overview

**PlantSAM-App** is an expert-guided, semi-automatic annotation tool built on top of the [PlantSAM2](https://github.com/IA-E-Col/PlantSAM) segmentation pipeline. It addresses the limitations of fully automatic segmentation by enabling experts to refine masks interactively using point prompts. This correction interface allows the transformation of unusable or incomplete segmentation masks into usable, high-quality annotations.

The tool is designed for:
- **Correcting segmentation errors** in difficult herbarium images
- **Expanding training datasets** for SAM2 fine-tuning
- **Improving accuracy** in trait analysis and downstream tasks

## Key Features

- Automatic pre-segmentation using the PlantSAM2 pipeline
- Interactive correction of masks using **point prompts**
- Image-by-image refinement workflow
- Export of corrected masks for retraining or evaluation
- Lightweight interface (Streamlit-based)

## Contributors
-  Youcef Sklab — Lead design & integration
-  Adam  Boukheddami — Developer
  
