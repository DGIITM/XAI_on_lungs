"""
Survey CAM quality across all 14 NIH pathology classes.

For each of the 14 pathologies, picks the 5 lowest-scoring "No Finding"
(normal) images and the 5 highest-scoring true-positive (abnormal) images,
then generates the full multi-technique CAM comparison for each — so we can
see, empirically, which pathology this model is actually well-calibrated on
(rather than assuming Pneumonia, which turned out to top out at 0.19).

Output: lung_cam_results_all14/{Pathology}/{normal,abnormal}/*_comparison.jpg
"""
import os
import json
import torch

from xai_lib.wrappers import get_cam_generator
from xai_lib.lung_model import load_lung_model, preprocess_xray_image
from generate_lung_cams import generate_and_save_comparison, TECHNIQUES

SCRATCH = r"C:\Users\user2\AppData\Local\Temp\claude\C--Users-user2-Desktop-xai-experimentation\9c843e86-d5b3-4ab1-8cc5-974dd66ba09f\scratchpad"
IMG_DIR = os.path.join(SCRATCH, "nih_all14")
OUTPUT_DIR = r"C:\Users\user2\Desktop\xai_experimentation\lung_cam_results_all14"

with open(os.path.join(SCRATCH, "all14_targets.json")) as f:
    targets = json.load(f)  # filename -> list of tags like "POS::Cardiomegaly" or "NEG"

CLASSES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema',
           'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
           'Mass', 'Hernia']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model, target_layers, class_names, pathology_index = load_lung_model('nih', device)

print("\nInitializing CAM generators (once, reused across all classes)...")
cam_generators = {}
for technique in TECHNIQUES:
    try:
        cam_generators[technique] = get_cam_generator(technique, model, target_layers, multi_label=True)
        print(f"  OK {technique}")
    except Exception as e:
        print(f"  FAIL {technique}: {e}")
techniques_available = list(cam_generators.keys())

# --- Score every fetched image once for all 14 pathologies ---
print(f"\nScoring {len(targets)} images for all 14 pathologies...")
scores = {}  # filename -> {class_name: prob}
with torch.no_grad():
    for i, fname in enumerate(sorted(targets.keys())):
        path = os.path.join(IMG_DIR, fname)
        input_tensor, _ = preprocess_xray_image(path, device)
        probs = torch.sigmoid(model(input_tensor))[0]
        scores[fname] = {c: probs[pathology_index[c]].item() for c in CLASSES}
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(targets)}")

# --- Per class: pick bottom-5 NEG and top-5 POS, generate CAMs ---
summary = []
for class_name in CLASSES:
    pos_files = [f for f, tags in targets.items() if f'POS::{class_name}' in tags]
    neg_files = [f for f, tags in targets.items() if 'NEG' in tags]

    pos_sorted = sorted(pos_files, key=lambda f: -scores[f][class_name])
    neg_sorted = sorted(neg_files, key=lambda f: scores[f][class_name])

    top5_pos = pos_sorted[:5]
    bottom5_neg = neg_sorted[:5]

    target_idx = pathology_index[class_name]
    print(f"\n=== {class_name} === (target idx {target_idx})", flush=True)
    print(f"  abnormal (top-5 of {len(pos_files)}): " +
          ", ".join(f"{f}={scores[f][class_name]:.3f}" for f in top5_pos), flush=True)
    print(f"  normal   (bottom-5 of {len(neg_files)}): " +
          ", ".join(f"{f}={scores[f][class_name]:.3f}" for f in bottom5_neg), flush=True)

    class_out = os.path.join(OUTPUT_DIR, class_name)
    already_done = (os.path.isdir(os.path.join(class_out, 'abnormal')) and
                     len(os.listdir(os.path.join(class_out, 'abnormal'))) >= 5 and
                     os.path.isdir(os.path.join(class_out, 'normal')) and
                     len(os.listdir(os.path.join(class_out, 'normal'))) >= 5)
    if already_done:
        print(f"  (already generated, skipping)", flush=True)
        summary.append({
            'class': class_name,
            'abnormal_max': max(scores[f][class_name] for f in top5_pos),
            'abnormal_min_of_top5': min(scores[f][class_name] for f in top5_pos),
            'normal_max_of_bottom5': max(scores[f][class_name] for f in bottom5_neg),
            'normal_min': min(scores[f][class_name] for f in bottom5_neg),
        })
        continue

    for fname in top5_pos:
        generate_and_save_comparison(
            os.path.join(IMG_DIR, fname), 'abnormal', model, target_layers,
            techniques_available, cam_generators, device,
            os.path.join(OUTPUT_DIR, class_name), target_idx, class_names, dpi=150)
    for fname in bottom5_neg:
        generate_and_save_comparison(
            os.path.join(IMG_DIR, fname), 'normal', model, target_layers,
            techniques_available, cam_generators, device,
            os.path.join(OUTPUT_DIR, class_name), target_idx, class_names, dpi=150)

    summary.append({
        'class': class_name,
        'abnormal_max': max(scores[f][class_name] for f in top5_pos),
        'abnormal_min_of_top5': min(scores[f][class_name] for f in top5_pos),
        'normal_max_of_bottom5': max(scores[f][class_name] for f in bottom5_neg),
        'normal_min': min(scores[f][class_name] for f in bottom5_neg),
    })

print("\n\n=== SUMMARY (sorted by abnormal_max, best-calibrated first) ===")
summary.sort(key=lambda s: -s['abnormal_max'])
for s in summary:
    print(f"  {s['class']:20s}  abnormal top-5: {s['abnormal_min_of_top5']:.3f}-{s['abnormal_max']:.3f}   "
          f"normal bottom-5: {s['normal_min']:.3f}-{s['normal_max_of_bottom5']:.3f}")

with open(os.path.join(SCRATCH, 'all14_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. Results in {OUTPUT_DIR}")
