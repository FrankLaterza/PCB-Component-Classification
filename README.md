# Getting started

Here's a comprehensive README for your SAM-based object extraction project:

# SAM Object Extraction Project

This project leverages Meta's Segment Anything Model (SAM) to automatically identify and extract objects from images. The pipeline detects objects using state-of-the-art segmentation and saves individual cropped objects to disk.

## Features
- Automatic object segmentation with SAM
- GPU acceleration support
- Batch processing-ready architecture
- Bounding box extraction with object isolation

## Prerequisites
- Python 3.8+
- CUDA 11.7+ (recommended for GPU acceleration)
- NVIDIA GPU with compatible drivers (optional but recommended)

## Installation
```bash
# Create virtual environment
python -m venv sam-env
source sam-env/bin/activate  # Linux/Mac
# sam-env\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python-headless segment-anything
```

## Getting Started
1. **Download SAM Checkpoint:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

2. **Directory Structure:**
```
project-root/
├── sam_vit_h_4b8939.pth
├── input.png
├── obj/
└── extract_objects.py
```

3. **Run Extraction:**
```bash
python extract_objects.py
```

## Usage Instructions
1. Place your input image in the project root as `input.png`
2. Execute the script
3. Find extracted objects in `/obj` directory with filenames:
   - `object_0.png`
   - `object_1.png`
   - ...

## Configuration Options
Modify these parameters in `extract_objects.py`:
```python
# For different SAM models
sam_model_registry["vit_l"]  # Large model
sam_model_registry["vit_b"]  # Base model

# Mask generator customization
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92
)
```

## Troubleshooting
**CUDA Issues:**
```bash
# For CPU-only operation:
sed -i 's/DEVICE = "cuda"/DEVICE = "cpu"/' extract_objects.py
```

**Common Fixes:**
1. Verify checkpoint file location
2. Ensure OpenCV can read input images:
   ```python
   print(cv2.imread("input.png") is None)  # Should print False
   ```
3. Confirm directory permissions for `/obj`

## Performance Notes
- VRAM Requirements:
  - ViT-H: ~7GB
  - ViT-L: ~5GB 
  - ViT-B: ~3GB
- Processing Time (512x512 image):
  - GPU: 200-400ms
  - CPU: 8-12 seconds

## Citation
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## License
This project uses the [Apache 2.0 license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) from the original SAM repository.

---
Answer from Perplexity: pplx.ai/share