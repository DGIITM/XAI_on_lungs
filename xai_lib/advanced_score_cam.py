import torch
import torch.nn.functional as f
import numpy as np

class AdvancedCAM(object):

    """
        AdvancedCAM
    """

    def __init__(self, model, target_layer, multi_label=False):
        self.model_arch = model
        self.activations = dict()
        self.gradients = dict()
        self.multi_label = multi_label
        self.model_arch.eval()
        
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        self.predicted_confidence_cam = 0
        self.predicted_confidence_cam_list = []
        self.avg_drop = 0
        self.avg_increase = 0

    def gaussian_filter(self, kernel_size=5, sigma=1):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                (- (x - (kernel_size - 1) / 2) ** 2 - (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)
            ),
            (kernel_size, kernel_size)
        )
        return kernel / np.sum(kernel)

    def apply_gaussian_smoothing(self, tensor, kernel_size=5, sigma=1):
        kernel = self.gaussian_filter(kernel_size, sigma)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(tensor.device)
        tensor = tensor.unsqueeze(0)
        smoothed_tensor = f.conv2d(tensor, kernel, padding=kernel_size // 2)
        return smoothed_tensor.squeeze()

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx]).to(input.device)
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        gradients = self.gradients['value']
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w)).to(input.device)

        with torch.no_grad():
            for i in range(k):
                # Generate saliency maps at different resolutions
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map_A = f.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                saliency_map_B = f.interpolate(saliency_map, size=(h // 16, w // 16), mode='bilinear', align_corners=False)
                saliency_map_C = f.interpolate(saliency_map_B, size=(h // 8, w // 8), mode='bilinear', align_corners=False)
                saliency_map_D = f.interpolate(saliency_map_C, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
                saliency_map_F = f.interpolate(saliency_map_D, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
                saliency_map_A1 = f.interpolate(saliency_map_F, size=(h, w), mode='bilinear', align_corners=False)
                
                # Apply Gaussian smoothing
                smoothed_saliency_map_A = self.apply_gaussian_smoothing(saliency_map_A.squeeze(), kernel_size=3, sigma=1)
                smoothed_saliency_map_A1 = self.apply_gaussian_smoothing(saliency_map_A1.squeeze(), kernel_size=3, sigma=1)

                # Compute element-wise maximum
                saliency_map_max = torch.max(smoothed_saliency_map_A, smoothed_saliency_map_A1)

                saliency_map = torch.tanh(saliency_map_max)
                norm_saliency_map = saliency_map.clone()
                output = self.model_arch(input * norm_saliency_map)

                output = torch.sigmoid(output) if self.multi_label else f.softmax(output, dim=1)
                score = output[0][predicted_class]
                score_saliency_map += score * saliency_map
                score_saliency_map = f.relu(score_saliency_map)

            score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map
            

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

    def metrics(self,model,input_):
        predicted_confidence, predicted_class = model(input_).max(1)
        predicted_class = predicted_class.item()
        predicted_confidence = predicted_confidence.item()

        scorecam_map = self.forward(input_)
        if scorecam_map == None:
            self.predicted_confidence_cam = 0
        else:
            scorecam_map_  = scorecam_map * input_
            scorecam_map_output = model(scorecam_map_)
            predicted_confidence_cam = scorecam_map_output[0][predicted_class]
            self.predicted_confidence_cam = predicted_confidence_cam.item()

        self.predicted_confidence_cam_list.append(self.predicted_confidence_cam)
        if predicted_confidence > self.predicted_confidence_cam:
          self.avg_drop = self.avg_drop + (predicted_confidence - self.predicted_confidence_cam)/predicted_confidence
        else:
          self.avg_increase = self.avg_increase + 1

    def percentize(self, count):
        self.avg_drop = self.avg_drop*100/count
        self.avg_increase = self.avg_increase*100/count