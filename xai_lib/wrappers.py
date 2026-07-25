import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from .reciprocam import ReciproCAM
from .advanced_score_cam import AdvancedCAM
from .rdelta_cam import RDeltaCAM

class BaseCAM:
    def __init__(self, model, target_layers, use_cuda=False):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda

    def generate(self, input_tensor, target_category=None):
        raise NotImplementedError

class GradCAMWrapper(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False):
        super().__init__(model, target_layers, use_cuda)
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor, target_category=None):
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

class GradCAMPlusPlusWrapper(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)
        self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    def generate(self, input_tensor, target_category=None):
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

class ScoreCAMWrapper(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)
        self.cam = ScoreCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor, target_category=None):
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

class ReciproCAMWrapper(BaseCAM):
    def __init__(self, model, target_layers, multi_label=False):
        super().__init__(model, target_layers)
        device = next(model.parameters()).device
        activation = 'sigmoid' if multi_label else 'softmax'
        # Hook the same layer the caller chose (e.g. model.layer4[-1]) so this technique
        # is explaining the same layer as GradCAM/ScoreCAM/etc. for a fair comparison.
        self.cam = ReciproCAM(model, device, target_layer=target_layers[-1], activation=activation)

    def generate(self, input_tensor, target_category=None):
        # ReciproCAM.__call__ returns (cam, index)
        # It expects input_tensor.
        cam, _ = self.cam(input_tensor, index=target_category)
        
        # The output cam is a tensor on the device, we need to convert to numpy
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
            
        # ReciproCAM returns (H_feat, W_feat) e.g. (7, 7)
        # We need to resize it to (H_input, W_input) e.g. (224, 224)
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (input_w, input_h))
        
        # Ensure it's float32
        return cam.astype(np.float32)

class AdvancedScoreCAMWrapper(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, multi_label=False):
        super().__init__(model, target_layers, use_cuda)
        self.cam = AdvancedCAM(model, target_layers[0], multi_label=multi_label)

    def generate(self, input_tensor, target_category=None):
        cam = self.cam(input_tensor, class_idx=target_category)
        
        if cam is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
            
        # AdvancedCAM returns (1, 1, H, W)
        if cam.ndim == 4:
            cam = cam[0, 0]
        elif cam.ndim == 3:
            cam = cam[0]
            
        # Resize to match input image (H, W)
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (input_w, input_h))
        
        # Ensure it's float32
        return cam.astype(np.float32)


class ScoreCAMPPWrapper(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False):
        super().__init__(model, target_layers, use_cuda)
        # TODO: Implement ScoreCAM++
        print("ScoreCAM++ is not yet implemented.")

    def generate(self, input_tensor, target_category=None):
         # Return a dummy CAM
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

class RDeltaCAMWrapper(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, multi_label=False):
        super().__init__(model, target_layers, use_cuda)
        device = next(model.parameters()).device
        activation = 'sigmoid' if multi_label else 'softmax'
        self.cam = RDeltaCAM(model, device, target_layer=target_layers[-1], activation=activation)

    def generate(self, input_tensor, target_category=None):
        cam, _ = self.cam(input_tensor, index=target_category)
        
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
            
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (input_w, input_h))
        
        return cam.astype(np.float32)

def get_cam_generator(name, model, target_layers, multi_label=False):
    if name == 'GradCAM':
        return GradCAMWrapper(model, target_layers)
    elif name == 'GradCAM++':
        return GradCAMPlusPlusWrapper(model, target_layers)
    elif name == 'ScoreCAM':
        return ScoreCAMWrapper(model, target_layers)
    elif name == 'ReciproCAM':
        return ReciproCAMWrapper(model, target_layers, multi_label=multi_label)
    elif name == 'ScoreCAM++':
        return ScoreCAMPPWrapper(model, target_layers)
    elif name == 'AdvancedCAM'  :
        return AdvancedScoreCAMWrapper(model, target_layers, multi_label=multi_label)
    elif name == 'RDeltaCAM'  :
        return RDeltaCAMWrapper(model, target_layers, multi_label=multi_label)
    else:
        raise ValueError(f"Unknown CAM technique: {name}")
