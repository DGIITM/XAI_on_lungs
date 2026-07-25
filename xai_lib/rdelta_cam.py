import copy
import torch

class RDeltaCAM:
    '''
    ReciproCam class contains official implementation of Reciprocal CAM algorithm for CNN architecture 
    published at CVPR2024 XAI4CV workshop.
    '''

    def __init__(self, model, device, target_layer=None, activation='softmax'):
        '''
        Creator of RDeltaCAM class

        Args:
            model: CNN architectur pytorch model
            device: runtime device type (ex, 'cuda', 'cpu')
            target_layer: the exact module (belonging to `model`) to hook the CAM on.
                If None, falls back to auto-detecting the last Conv2d layer.
                Passing this explicitly keeps the hooked layer identical to whatever
                other techniques (GradCAM, ScoreCAM, ...) are using for a fair comparison.
            activation: 'softmax' for single-label classifiers, 'sigmoid' for multi-label
                (e.g. multi-pathology chest X-ray) models.
        '''

        # Resolve the target layer's name in the *original* model before deep-copying,
        # so we can find the equivalent module inside the copy (hooks on the original
        # module object would not affect the copy used for inference).
        target_name = None
        if target_layer is not None:
            for name, module in model.named_modules():
                if module is target_layer:
                    target_name = name
                    break
            if target_name is None:
                raise ValueError("target_layer must be a submodule of model")

        self.model = copy.deepcopy(model)
        self.model.eval()
        self.device = device
        self.feature = None
        self.activate = torch.sigmoid if activation == 'sigmoid' else torch.nn.Softmax(dim=1)
        self.target_layers = []
        self.conv_depth = 0

        if target_name is not None:
            hook_layer = dict(self.model.named_modules())[target_name]
        else:
            self._find_target_layer(self.model)
            hook_layer = self.target_layers[-1]
        hook_layer.register_forward_hook(self._cam_hook())


    def _find_target_layer(self, m, depth=0):
        '''
        Searching target layer by name from given network model as recursive manner.
        '''

        children = dict(m.named_children())
        if not children:
            if isinstance(m, torch.nn.Conv2d):
                self.target_layers.clear()
                self.target_layers.append(m)
                self.conv_depth = depth
            elif self.conv_depth == depth and any(self.target_layers) and isinstance(m, torch.nn.BatchNorm2d):
                self.target_layers.append(m)
            elif self.conv_depth == depth and any(self.target_layers) and isinstance(m, torch.nn.ReLU):
                self.target_layers.append(m)
        else:
            for name, child in children.items():
                self._find_target_layer(child, depth+1)


    def _cam_hook(self):
        '''
        Setup hook funtion for generating new masked features for calculating reciprocal activation score 
        '''

        def fn(_, input, output):
            self.feature = output[0].unsqueeze(0)
            bs, nc, h, w = self.feature.shape
            retain_features = self._mosaic_feature(self.feature, nc, h, w, mode='retain')
            suppress_features = self._mosaic_feature(self.feature, nc, h, w, mode='suppress')
            all_features = torch.cat([self.feature, retain_features, suppress_features], dim=0)
            return all_features

        return fn


    def _mosaic_feature(self, feature_map, nc, h, w, mode='retain'):
        feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
        mask = torch.ones(h * w, nc, h, w).to(self.device)

        spatial_order = torch.arange(h * w).reshape(h, w)
        for i in range(h):
            for j in range(w):
                k = spatial_order[i, j]
                if mode == 'retain':
                    mask[k] = 0
                    mask[k, :, i, j] = 1.0  # only retain (i,j)
                elif mode == 'suppress':
                    mask[k, :, i, j] = 0.0  # suppress only (i,j)

        return feature_map_repeated * mask


    def _get_class_activaton_map(self, predictions, index, h, w):
       # predictions: shape [1 + 2hw, num_classes]
        retain_preds = predictions[1:h*w+1, index]
        suppress_preds = predictions[h*w+1:, index]
        cam = retain_preds - suppress_preds
        cam = cam.reshape(h, w)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam


    def __call__(self, input, index=None):
        with torch.no_grad():
            BS, _, _, _ = input.shape
            if BS > 1:
                raise ValueError("RDeltaCAM currently supports only one image at a time.")

            predictions = self.model(input)
            predictions = self.activate(predictions)

            if index is None:
                index = predictions[0].argmax().item()

            _, _, h, w = self.feature.shape
            cam = self._get_class_activaton_map(predictions, index, h, w)


        return cam, index

