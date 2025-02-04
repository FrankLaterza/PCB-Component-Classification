import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize SAM with GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

# 1. Create SAM model first
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)  # Remove weights_only
sam.to(device=DEVICE)

# 2. Then create mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Rest of your code remains the same
image = cv2.cvtColor(cv2.imread("input.png"), cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

# Extract and save objects
for i, mask in enumerate(masks):
    x1, y1 = mask["bbox"][0], mask["bbox"][1]
    x2, y2 = x1 + mask["bbox"][2], y1 + mask["bbox"][3]
    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(f"obj/object_{i}.png", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
