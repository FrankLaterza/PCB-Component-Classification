# PCB Component Classification
This deep learning project is designed to take an image and break it down into its components and classify them. We can create an image with boxes and labels demonstrate the results.

This project uses a two stage process to output its results. The first is Meta's SAM model which is designed to segment the components into smaller pictures. The second model is a CNN thats trained on the [FICS-PCB](https://trust-hub.org/#/data/fics-pcb) dataset created by the SCAN lab at UF. This data set has pictures of PCBs and segments images labeled by their correct names. This model will be used on the output of the SAM to correctly label the components.

## How to use
The main entry point is the src/ai.py image files that takes arguments to run the models

```cmd
# train the cnn with the files under the directory dataset/mega
python src/ai.py -train NUM-OF-EPOCHS

# test the classification model on a test image
python src/ai.py -classify PATH/TO/IMAGE

# populate the objs directory with segmented images a PCB image
python src/ai.py -segment PATH/TO/IMAGE

# segment and classify a PCB image
python src/ai.py -full PATH/TO/IMAGE
```

## Installation Guide (Windows)

1. **Python 3.11** (required for PyTorch CUDA support)  
   [Download from Microsoft Store](https://apps.microsoft.com/detail/python-311/9NRWMJP3RSLX?hl=en-us&gl=US)  
   *Check "Add Python to PATH" during installation*

2. **CUDA 12.1 Toolkit**  
   [Download CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)  
   Verify installation:
   ```cmd
   nvcc --version
   ```

3. **NVIDIA Drivers**  
   Ensure latest drivers for your GPU:  
   [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx)

### Package Installation
```bash
# Create virtual environment
python -m venv sam_env
sam_env\Scripts\activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SAM and dependencies
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python matplotlib
```