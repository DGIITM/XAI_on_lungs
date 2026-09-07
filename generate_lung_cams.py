"""
Batch CAM generation script for Lung Disease dataset.

Generates side-by-side comparison figures showing CAMs from all XAI techniques
for each image in the dataset. Designed for qualitative analysis in research papers.

Usage:
    # Generate CAMs for test set, all techniques, save to results/
    python generate_lung_cams.py --dataset chest_xray/test --output results/ --use-cuda

    # Only specific techniques
    python generate_lung_cams.py --dataset chest_xray/test --output results/ --techniques GradCAM RDeltaCAM --use-cuda

    # Limit number of images
    python generate_lung_cams.py --dataset chest_xray/test --output results/ --num-images 10 --use-cuda
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from xai_lib.wrappers import get_cam_generator
from xai_lib.lung_model import load_lung_model, preprocess_xray_image


TECHNIQUES = ['GradCAM', 'GradCAM++', 'ScoreCAM', 'ReciproCAM', 'AdvancedCAM', 'RDeltaCAM']
CLASS_NAMES_KAGGLE = ['NORMAL', 'PNEUMONIA']


def get_args():
    parser = argparse.ArgumentParser(description='Lung Disease CAM Generation')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (ImageFolder: class1/, class2/ subdirs)')
    parser.add_argument('--output', type=str, default='lung_cam_results',
                        help='Output directory to save CAM visualizations')
    parser.add_argument('--techniques', nargs='+', default=TECHNIQUES,
                        choices=TECHNIQUES,
                        help='XAI techniques to use (default: all)')
    parser.add_argument('--weights', type=str, default='chex',
                        choices=['nih', 'chex', 'rsna', 'all'],
                        help='TorchXRayVision pretrained weights to use')
    parser.add_argument('--num-images', type=int, default=20,
                        help='Number of images to process (default: 20)')
    parser.add_argument('--images-per-class', type=int, default=None,
                        help='Number of images per class (overrides --num-images if set)')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--target-class', type=str, default='Pneumonia',
                        help="Target pathology for CAM (e.g., Pneumonia). Default: 'Pneumonia'. "
                             "Pass an empty string to fall back to the model's top prediction instead.")
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures')
    return parser.parse_args()


def xray_preprocess(image_path):
    """
    Load and preprocess a chest X-ray for TorchXRayVision.
    Returns input_tensor (1, 1, 224, 224) and rgb_img (224, 224, 3) for overlay.

    Delegates to the single shared implementation in xai_lib.lung_model so the
    input a CAM sees here is byte-identical to the single-image path in main.py.
    """
    return preprocess_xray_image(image_path, device='cpu')


def overlay_cam_on_image(rgb_img, cam):
    """Overlay a heatmap CAM on an RGB image."""
    cam_normalized = np.clip(cam, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * rgb_img + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


def generate_and_save_comparison(image_path, true_label, model, target_layers,
                                  techniques, cam_generators, device, output_dir,
                                  target_class_idx, class_names, dpi=150):
    """
    Generate a side-by-side comparison figure for a single image.
    """
    # Preprocess
    input_tensor, rgb_img = xray_preprocess(image_path)
    input_tensor = input_tensor.to(device)

    # Get model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0]  # Multi-label sigmoid
        pred_idx = probs.argmax().item()
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) and class_names[pred_idx] else f"class_{pred_idx}"
        pred_conf = probs[pred_idx].item()

    # Use specified target class or top prediction
    cam_target = target_class_idx if target_class_idx is not None else pred_idx
    # Probability of the class actually being explained by the CAM below (may differ
    # from the model's own top prediction above when --target-class is set).
    target_name = class_names[cam_target] if cam_target < len(class_names) and class_names[cam_target] else f"class_{cam_target}"
    target_conf = probs[cam_target].item()

    # --- Plot ---
    n_cols = len(techniques) + 1  # +1 for original
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # Original image
    axes[0].imshow(rgb_img)
    axes[0].set_title(f'Original\nTrue: {true_label}\nTop pred: {pred_class} ({pred_conf:.2f})\n'
                       f'{target_name} (explained): {target_conf:.2f}',
                       color='white', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    # CAM overlays
    for i, technique in enumerate(techniques):
        ax = axes[i + 1]
        try:
            with torch.enable_grad():
                cam = cam_generators[technique].generate(input_tensor, target_category=cam_target)
            overlay = overlay_cam_on_image(rgb_img, cam)
            ax.imshow(overlay)
            ax.set_title(technique, color='white', fontsize=9, fontweight='bold')
        except Exception as e:
            ax.imshow(rgb_img)
            ax.set_title(f'{technique}\n(Error)', color='red', fontsize=8)
            print(f"  Warning: {technique} failed: {e}")
        ax.axis('off')

    # Filename as suptitle
    img_name = Path(image_path).stem
    fig.suptitle(f'{img_name} | Target: {target_name} ({target_conf:.2f})',
                  color='white', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.0)

    # Save
    save_path = os.path.join(output_dir, true_label, f'{img_name}_comparison.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)

    return save_path


def main():
    args = get_args()
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, target_layers, class_names, pathology_index = load_lung_model(args.weights, device)

    # Resolve target class index
    target_class_idx = None
    if args.target_class:
        target_class_idx = pathology_index.get(args.target_class)
        if target_class_idx is None:
            print(f"Warning: Target class '{args.target_class}' not found. Available: {class_names}")
        else:
            print(f"Target class: {args.target_class} (index {target_class_idx})")

    # Initialize all CAM generators ONCE (expensive for some like ScoreCAM)
    print("\nInitializing CAM generators...")
    cam_generators = {}
    for technique in args.techniques:
        try:
            cam_generators[technique] = get_cam_generator(technique, model, target_layers, multi_label=True)
            print(f"  ✓ {technique}")
        except Exception as e:
            print(f"  ✗ {technique}: {e}")

    if not cam_generators:
        print("No CAM generators initialized. Exiting.")
        return

    techniques_available = list(cam_generators.keys())

    # Collect images per class
    dataset_path = args.dataset
    class_dirs = sorted([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))
                          and not d.startswith('.')])
    print(f"\nFound classes: {class_dirs}")

    image_paths = []  # List of (image_path, class_label)
    for class_label in class_dirs:
        class_dir = os.path.join(dataset_path, class_label)
        img_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                     and not f.startswith('.')]

        if args.images_per_class:
            img_files = img_files[:args.images_per_class]

        for img_file in img_files:
            image_paths.append((os.path.join(class_dir, img_file), class_label))

    # Limit total number of images if images_per_class not set
    if not args.images_per_class:
        image_paths = image_paths[:args.num_images]

    print(f"\nProcessing {len(image_paths)} images...")
    os.makedirs(args.output, exist_ok=True)

    # Generate CAMs
    success_count = 0
    for idx, (image_path, true_label) in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] {true_label}/{Path(image_path).name}")
        try:
            save_path = generate_and_save_comparison(
                image_path, true_label, model, target_layers,
                techniques_available, cam_generators, device, args.output,
                target_class_idx, class_names, args.dpi
            )
            print(f"  Saved: {save_path}")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! Generated {success_count}/{len(image_paths)} comparison figures.")
    print(f"Results saved to: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
