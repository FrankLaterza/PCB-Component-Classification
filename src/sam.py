import time
import cv2
import torch

def run_sam(file_path) :
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # hardware configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    # model initialization
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # image loading and conversion
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    boxed_image = image.copy()  # create copy for visualization

    # core processing with timing
    start_time = time.time()
    masks = mask_generator.generate(image)
    processing_time = time.time() - start_time

    # bounding box visualization
    for mask in masks:
        x, y, w, h = mask["bbox"]
        cv2.rectangle(
            boxed_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # green color in rgb
            2  # line thickness
        )

    # save outputs
    cv2.imwrite("boxed_objects.png", cv2.cvtColor(boxed_image, cv2.COLOR_RGB2BGR))

    # object extraction
    for i, mask in enumerate(masks):
        x1, y1 = mask["bbox"][0], mask["bbox"][1]
        x2, y2 = x1 + mask["bbox"][2], y1 + mask["bbox"][3]
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(f"obj/object_{i}.png", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # performance summary
    print(f"\nPROCESSING SUMMARY")
    print(f"total time: {processing_time:.2f} seconds")
    print(f"objects detected: {len(masks)}")
    print(f"time per object: {processing_time/len(masks):.4f}s" if len(masks) > 0 else "No objects detected")
    print(f"output saved: boxed_objects.png and {len(masks)} objects in /obj")