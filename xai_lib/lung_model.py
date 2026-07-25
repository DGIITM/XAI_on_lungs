"""
Lung Disease Model Loader using TorchXRayVision (CheXNet / DenseNet-121).

TorchXRayVision provides pretrained DenseNet-121 models on multiple chest X-ray datasets.
Key difference from standard ImageNet models:
  - Input: Grayscale (1 channel), float32, normalized to [-1024, 1024]
  - Output: Multi-label predictions over disease classes

Usage:
    from xai_lib.lung_model import load_lung_model
    model, target_layers, class_names, preprocess = load_lung_model(weights='densenet121-res224-nih', device='cuda')
"""

import torch
import torchvision.transforms as transforms
import numpy as np


# Available TorchXRayVision weights
AVAILABLE_WEIGHTS = {
    'nih':     'densenet121-res224-nih',    # NIH ChestX-ray14, 14 diseases
    'chex':    'densenet121-res224-chex',   # CheXpert
    'rsna':    'densenet121-res224-rsna',   # RSNA Pneumonia Challenge
    'all':     'densenet121-res224-all',    # All datasets combined, 18 pathologies
    'mimic_nb': 'densenet121-res224-mimic_nb', # MIMIC (no bias correction)
}

# NIH ChestX-ray14 class names (14 pathologies)
NIH_CLASS_NAMES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# Kaggle Chest X-Ray Pneumonia classes (binary, used by our fine-tuned model)
PNEUMONIA_CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def get_lung_preprocess():
    """
    Returns the preprocessing transform for TorchXRayVision models.
    
    TorchXRayVision normalize:
    - Converts to grayscale, scales to [0, 255], then normalizes to [-1024, 1024]
    - Resizes to 224x224
    """
    import torchxrayvision as xrv
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return transform


def _center_crop_square(img_2d):
    """Crop a (H, W) array to a centered square of side min(H, W).

    This matches TorchXRayVision's XRayCenterCrop. It is essential because the
    pretrained DenseNets were trained on center-cropped square X-rays: naively
    resizing a ~1.4:1 chest film straight to 224x224 squashes the anatomy
    horizontally, which both worsens the model's own predictions and shifts
    every CAM (the distortion is upstream of all techniques, so it degrades
    GradCAM, ScoreCAM, ReciproCAM, RDeltaCAM, ... equally).
    """
    h, w = img_2d.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_2d[y0:y0 + side, x0:x0 + side]


def preprocess_xray_image(image_path, device='cpu'):
    """
    Load and preprocess a chest X-ray image for TorchXRayVision models.

    Single source of truth for X-ray preprocessing: both the single-image entry
    point (main.py) and the batch comparison script (generate_lung_cams.py) call
    this, so the input a CAM sees is guaranteed identical regardless of path.

    Pipeline (matches TorchXRayVision): grayscale -> normalize to [-1024, 1024]
    -> center-crop to square -> resize to 224. The RGB visualization image is
    derived from the *same* center-cropped pixels so the heatmap overlay stays
    spatially aligned with the anatomy the model actually saw.

    Args:
        image_path: path to image (JPEG/PNG)
        device: 'cpu' or 'cuda'

    Returns:
        input_tensor: shape (1, 1, 224, 224), normalized to [-1024, 1024]
        rgb_img: shape (224, 224, 3), float32, [0, 1] for visualization
    """
    from PIL import Image
    import cv2

    # Load image in grayscale
    img = Image.open(image_path).convert('L')  # Grayscale
    img_np = np.array(img).astype(np.float32)  # (H, W)

    # TorchXRayVision normalization: scale to [-1024, 1024]
    img_np = (img_np / 255.0) * 2048 - 1024

    # Center-crop to square then resize (NOT a direct squash-resize)
    img_np = _center_crop_square(img_np)
    img_np = cv2.resize(img_np, (224, 224))

    # Shape: (1, 1, 224, 224)
    input_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # RGB visualization from the SAME preprocessed pixels so the overlay aligns
    vis = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    rgb_img = np.stack([vis] * 3, axis=-1).astype(np.float32)

    return input_tensor, rgb_img


def load_lung_model(weights='chex', device='cpu'):
    """
    Load a pretrained lung disease model from TorchXRayVision.
    
    Args:
        weights: one of 'nih', 'chex', 'rsna', 'all', 'mimic_nb'
        device: 'cpu' or 'cuda'
    
    Returns:
        model: pretrained DenseNet-121 model (eval mode)
        target_layers: list of target layers for CAM
        class_names: list of disease class names
        pathology_index: dict mapping class name to index
    """
    try:
        import torchxrayvision as xrv
    except ImportError:
        raise ImportError(
            "torchxrayvision is not installed.\n"
            "Install it with: pip install torchxrayvision"
        )
    
    weight_key = AVAILABLE_WEIGHTS.get(weights, weights)
    print(f"Loading TorchXRayVision model: {weight_key}")
    
    model = xrv.models.DenseNet(weights=weight_key)
    model.to(device)
    model.eval()

    # Wrap the model to provide a clean post-ReLU target layer.
    # DenseNet's dense-block output (and norm5) is pre-ReLU. Modifying it via a forward hook
    # breaks the BatchNorm statistics and calibration for the rest of the network, causing
    # RDeltaCAM and ReciproCAM to produce inverted results (because their masked features 
    # pass through norm5 which gets wildly shifted means).
    # By creating a clean post-ReLU layer, ScoreCAM gets the non-negative features it needs,
    # and RDeltaCAM/ReciproCAM can safely mask with 0 without passing through any more BatchNorms.
    class DenseNetWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.post_relu = torch.nn.ReLU()
            self.pathologies = base_model.pathologies
            
        def forward(self, x):
            features = self.base_model.features(x)
            out = self.post_relu(features)
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.base_model.classifier(out)
            return out
            
    wrapper_model = DenseNetWrapper(model)
    wrapper_model.eval()
    
    # Target layer is now the explicit post-ReLU layer we added
    target_layers = [wrapper_model.post_relu]
    
    # Class names from the model's pathologies list
    class_names = wrapper_model.pathologies
    pathology_index = {name: i for i, name in enumerate(class_names)}
    
    print(f"Model loaded. Pathologies: {list(class_names)}")

    return wrapper_model, target_layers, list(class_names), pathology_index
