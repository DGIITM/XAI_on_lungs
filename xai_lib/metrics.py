import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from PIL import Image
from sklearn.metrics import auc
from torchmetrics import PearsonCorrCoef
from .wrappers import get_cam_generator

MOD = 10

def average_drop_increase(model, data_loader, Height, Width, technique, target_layers, device='cpu', multi_label=False, target_class_idx=None):
    # Metric for average drop and increase
    avg_drop = 0.0
    avg_inc = 0.0

    N = 0

    eval_model = model
    eval_model.eval()
    activate = torch.sigmoid if multi_label else torch.nn.Softmax(dim=1)

    # Initialize CAM generator
    cam_generator = get_cam_generator(technique, model, target_layers, multi_label=multi_label)

    for batch_idx, (images, labels) in enumerate(data_loader, 0):
        images, labels = images.to(device), labels.to(device)
        current_batch_size = images.shape[0]
        C = images.shape[1]  # Dynamic channel count: 1 for grayscale, 3 for RGB

        with torch.no_grad():
            predictions = model(images)
            predictions = activate(predictions)

        # Resize explanation and tensors according to current_batch_size
        explanation = torch.zeros(current_batch_size, C, Height, Width).to(device)
        yc = torch.zeros(current_batch_size).to(device)
        oc = torch.zeros(current_batch_size).to(device)

        for i in range(current_batch_size):
            class_id = target_class_idx if target_class_idx is not None else labels[i].item()
            yc[i] = predictions[i][class_id]

            # Generate CAM using the wrapper
            # Wrapper expects (1, C, H, W) input
            # Ensure gradients are enabled for CAM generation
            with torch.enable_grad():
                cam = cam_generator.generate(images[i].unsqueeze(0), target_category=class_id)

            # Wrapper returns numpy array (H, W), convert to tensor
            cam = torch.from_numpy(cam).to(device)

            cam = to_pil_image(cam, mode='F')
            cam = cam.resize((Height,Width), resample=Image.Resampling.BICUBIC)
            cam = pil_to_tensor(cam)
            explanation[i] = torch.mul(images[i], cam.repeat(C,1,1).to(device))

        explanation = explanation.to(device)
        with torch.no_grad():
            o_predictions = eval_model(explanation)
            o_predictions = activate(o_predictions)
        
        for i in range(current_batch_size):
            class_id = target_class_idx if target_class_idx is not None else labels[i].item()
            oc[i] = o_predictions[i][class_id]
        drop = torch.nn.functional.relu(yc - oc) / yc
        inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
        avg_drop += drop.sum()
        avg_inc += inc

        N += current_batch_size

        if batch_idx % MOD == MOD-1:
            print('Batch ID: ', batch_idx + 1)

    avg_drop /= N * 0.01
    avg_inc /= N * 0.01

    return avg_drop, avg_inc


def dauc_iauc(model, data_loader, Height, Width, technique, target_layers, device='cpu', multi_label=False, target_class_idx=None):
    # Metric for Deletion/Insertion Area Under Curve (DAUC/IAUC)
    DAUC_score = 0.0
    IAUC_score = 0.0

    N = 0
    eval_model = model
    eval_model.eval()
    activate = torch.sigmoid if multi_label else torch.nn.Softmax(dim=1)

    cam_generator = get_cam_generator(technique, model, target_layers, multi_label=multi_label)

    for batch_idx, (images, labels) in enumerate(data_loader, 0):
        images, labels = images.to(device), labels.to(device)
        current_batch_size = images.shape[0]
        C = images.shape[1]  # Dynamic channel count
        h = int(Height/32)
        w = int(Width/32)

        with torch.no_grad():
            predictions = model(images)
            predictions = activate(predictions)

        dx = int(Width/w)
        dy = int(Height/h)
        ck = torch.zeros(h*w+1).to(device)
        for i in range(current_batch_size):
            class_id = target_class_idx if target_class_idx is not None else labels[i].item()

            with torch.enable_grad():
                cam = cam_generator.generate(images[i].unsqueeze(0), target_category=class_id)
            cam = torch.from_numpy(cam).to(device)

            # CAMs come back at input resolution (Height, Width); coarsen to the
            # (h, w) patch grid so we can rank and mask patches for deletion/insertion.
            cam_small = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear')[0, 0]

            cam_flat = cam_small.reshape(-1,)
            _, s_index = cam_flat.sort(dim=0, descending=True)

            del_mask = torch.ones(C, Height, Width).to(device)
            inc_mask = torch.zeros(C, Height, Width).to(device)
            base_color = torch.mean(images[i], dim=(1, 2)).unsqueeze(0)
            base_color = base_color.reshape(1,C,1,1)
            auc_images = base_color.repeat(2*h*w,1,Height, Width).to(device)
            for j in range(h*w):
                s_idx = int(s_index[j].cpu().item())
                ci = s_idx//w
                cj = s_idx - ci*w
                xs = int(cj*dx)
                xe = int(min((cj+1)*dx, Width-1))
                ys = int(ci*dy)
                ye = int(min((ci+1)*dy, Height-1))
                del_mask[:,ys:ye+1,xs:xe+1] = 0.0
                inc_mask[:,ys:ye+1,xs:xe+1] = 1.0
                auc_images[j] = images[i]*del_mask
                auc_images[h*w + j] = auc_images[h*w +j]*del_mask + images[i]*inc_mask
            
            with torch.no_grad():
                o_predictions = eval_model(auc_images)
                o_predictions = activate(o_predictions)
            
            ck[0] = predictions[i][class_id]
            ck[1:] = o_predictions[:h*w,class_id]
            x = np.arange(0, len(ck))
            y = ck.detach().cpu().numpy()
            DAUC_score += auc(x, y) / len(ck)
            ck[h*w] = predictions[i][class_id]
            ck[:h*w] = o_predictions[h*w:,class_id]
            y = ck.detach().cpu().numpy()
            IAUC_score += auc(x, y) / len(ck)
        N += current_batch_size

        if batch_idx % MOD == MOD-1:
            print('Batch ID: ', batch_idx + 1)

    DAUC_score = DAUC_score / (N * 0.01)
    IAUC_score = IAUC_score / (N * 0.01)

    return DAUC_score, IAUC_score


def ADCC(model, data_loader, Height, Width, technique, target_layers, device='cpu', multi_label=False, target_class_idx=None):
    # Metric for Average Drop, Increase, Coherency, Complexity (ADCC)
    adcc = 0.0
    coherency = 0.0
    complexity = 0.0
    avg_drop = 0.0
    avg_inc = 0.0

    pearson = PearsonCorrCoef().to(device)

    N = 0
    eval_model = model
    eval_model.eval()
    activate = torch.sigmoid if multi_label else torch.nn.Softmax(dim=1)

    cam_generator = get_cam_generator(technique, model, target_layers, multi_label=multi_label)

    for batch_idx, (images, labels) in enumerate(data_loader, 0):
        images, labels = images.to(device), labels.to(device)
        current_batch_size = images.shape[0]
        C = images.shape[1]  # Dynamic channel count

        h = int(Height/32)
        w = int(Width/32)

        with torch.no_grad():
            predictions = model(images)
            predictions = activate(predictions)

        pre_cam = torch.zeros(current_batch_size, h*w).to(device)
        explanation = torch.zeros(current_batch_size, C, Height, Width).to(device)
        yc = torch.zeros(current_batch_size).to(device)
        oc = torch.zeros(current_batch_size).to(device)

        for i in range(current_batch_size):
            class_id = target_class_idx if target_class_idx is not None else labels[i].item()
            yc[i] = predictions[i][class_id]

            with torch.enable_grad():
                cam = cam_generator.generate(images[i].unsqueeze(0), target_category=class_id)
            cam = torch.from_numpy(cam).to(device)

            # Resize for complexity/coherency calculation (h*w)
            cam_small = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear')[0, 0]
            
            pre_cam[i] = cam_small.reshape(-1,).to(device)
            complexity += cam_small.sum()/(h*w)
            
            cam_pil = to_pil_image(cam, mode='F')
            cam_pil = cam_pil.resize((Height,Width), resample=Image.Resampling.BICUBIC)
            cam_tensor = pil_to_tensor(cam_pil)
            explanation[i] = torch.mul(images[i], cam_tensor.repeat(C,1,1).to(device))

        with torch.no_grad():
            o_predictions = eval_model(explanation)
            o_predictions = activate(o_predictions)
            
        for i in range(current_batch_size):
            class_id = target_class_idx if target_class_idx is not None else labels[i].item()
            oc[i] = o_predictions[i][class_id]

            # Generate CAM on explanation
            with torch.enable_grad():
                cam = cam_generator.generate(explanation[i].unsqueeze(0), target_category=class_id)
            cam = torch.from_numpy(cam).to(device)
            cam_small = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear')[0, 0]
            
            cam_flat = cam_small.reshape(-1,).to(device)
            coherency += 0.5 * (1.0 + pearson(cam_flat, pre_cam[i]))
            
        drop = torch.nn.functional.relu(yc - oc) / yc
        inc = torch.count_nonzero(torch.nn.functional.relu(oc - yc))
        avg_drop += drop.sum()
        avg_inc += inc

        N += current_batch_size

        if batch_idx % MOD == MOD-1:
            print('Batch ID: ', batch_idx + 1)

    avg_drop = avg_drop.cpu().item() / N
    avg_inc = avg_inc.cpu().item() / N
    coherency = coherency.cpu().item() / N
    complexity = complexity.cpu().item() / N
    
    # Avoid division by zero
    term1 = 1.0/coherency if coherency != 0 else 0
    term2 = 1.0/(1-complexity) if (1-complexity) != 0 else 0
    term3 = 1.0/(1-avg_drop) if (1-avg_drop) != 0 else 0
    
    adcc = 3.0/(term1 + term2 + term3)

    avg_drop *= 100.0
    avg_inc *= 100.0
    coherency *= 100.0
    complexity *= 100.0
    adcc *= 100.0

    return avg_drop, avg_inc, coherency, complexity, adcc
