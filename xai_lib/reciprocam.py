import copy
import torch

class ReciproCAM:
    '''
    ReciproCam class contains official implementation of Reciprocal CAM algorithm for CNN architecture 
    published at CVPR2024 XAI4CV workshop.
    '''

    def __init__(self, model, device, target_layer=None, activation='softmax'):
        '''
        Creator of ReciproCAM class

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
        filter = [[1/16.0, 1/8.0, 1/16.0],
                    [1/8.0, 1/4.0, 1/8.0],
                    [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filter).to(device)
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
            new_features = self._mosaic_feature(self.feature, nc, h, w, False)
            new_features = torch.cat((self.feature, new_features), dim = 0)
            return new_features

        return fn


    def _mosaic_feature(self, feature_map, nc, h, w, is_gaussian=False):
        '''
        Generate spatially masked feature map [h*w, nc, h, w] from input feature map [1, nc, h, w].
        If is_gaussian is true then the spatially masked feature map's value are filtered by 3x3 Gaussian filter.  
        '''
        
        if is_gaussian == False:
            feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, nc, h, w).to(self.device)
            spatial_order = torch.arange(h * w).reshape(h, w)
            for i in range(h):
                for j in range(w):
                    k = spatial_order[i, j]
                    mosaic_feature_map_mask[k, :, i, j] = torch.ones(nc).to(self.device)
            new_features = feature_map_repeated * mosaic_feature_map_mask
        else:
            new_features = torch.zeros(h*w, nc, h, w).to(self.device)
            for b in range(h*w):
                for i in range(h): #0...h-1
                    kx_s = max(i-1, 0)
                    kx_e = min(i+1, h-1)
                    if i == 0: sx_s = 1
                    else: sx_s = 0
                    if i == h-1: sx_e = 1
                    else: sx_e = 2
                    for j in range(w): #0...w-1
                        ky_s = max(j-1, 0)
                        ky_e = min(j+1, w-1)
                        if j == 0: sy_s = 1
                        else: sy_s = 0
                        if j == w-1: sy_e = 1
                        else: sy_e = 2
                        if b == i*w + j:
                            r_feature_map = feature_map[0,:,i,j].reshape(feature_map.shape[1],1,1)
                            r_feature_map = r_feature_map.repeat(1,3,3)
                            score_map = r_feature_map*self.gaussian.repeat(feature_map.shape[1],1,1)
                            new_features[b,:,kx_s:kx_e+1,ky_s:ky_e+1] = score_map[:,sx_s:sx_e+1,sy_s:sy_e+1]

        return new_features


    def _get_class_activaton_map(self, mosaic_predic, index, h, w):
        '''
        Calculate class activation map from the prediction result of mosaic feature input.
        '''
        
        cam = (mosaic_predic[:,index]).reshape((h, w))
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
                cam = []
                for b in range(BS):
                    prediction = self.model(input[b, : , :, :].unsqueeze(0))
                    prediction = self.activate(prediction)
                    if index == None:
                        index = prediction[0].argmax().item()
                    _, _, h, w = self.feature.shape
                    cam.append(self._get_class_activaton_map(prediction[1:, :], index, h, w))
            else:
                prediction = self.model(input)
                prediction = self.activate(prediction)
                if index == None:
                    index = prediction[0].argmax().item()
                _, _, h, w = self.feature.shape
                cam = self._get_class_activaton_map(prediction[1:, :], index, h, w)

        return cam, index