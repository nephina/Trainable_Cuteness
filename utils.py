import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from model import CNNSingleValueRanker


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())

def get_saliency_map(image, saliency_map):
    """ 
    Get saliency map from image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (512,512))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply colormap (STOP USING JET!!)
    # https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_TURBO)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    #img_with_heatmap = cv2.addWeighted(image,0.5,color_heatmap,0.5,0)

    return img_with_heatmap

class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition
    """

    def __init__(self, model, im_size = (3,64,64) ):
        self.model = model
        self.im_size = (1,) + im_size
        self.model_ext = FullGradExtractor(model, im_size)
        self.biases = self.model_ext.getBiases()


    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """

        self.model.eval()
        image = image.requires_grad_()
        out = self.model(image)
        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]
        
        # Select the output unit corresponding to the target class
        # -1 compensates for negation in nll_loss function
        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        input_gradient, feature_gradients = self.model_ext.getFeatureGrads(image, output_scalar)

        # Compute feature-gradients \times bias 
        bias_times_gradients = []
        L = len(self.biases)

        for i in range(L):

            # feature gradients are indexed backwards 
            # because of backprop
            g = feature_gradients[L-1-i]

            # reshape bias dimensionality to match gradients
            bias_size = [1] * len(g.size())
            bias_size[1] = self.biases[i].size(0)
            b = self.biases[i].view(tuple(bias_size))
            
            bias_times_gradients.append(g * b.expand_as(g))

        return input_gradient, bias_times_gradients

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def saliency(self, image, target_class=None):
        #FullGrad saliency

        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=target_class)

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        im_size = image.size()

        # Aggregate Bias-gradients
        for i in range(len(bias_grad)):

            # Select only Conv layers
            if len(bias_grad[i].size()) == len(im_size): 
                temp = self._postProcess(bias_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=True)
                cam += gradient.sum(1, keepdim=True)

        return cam

class FullGradExtractor:
    #Extract tensors needed for FullGrad using hooks
    
    def __init__(self, model, im_size = (3,224,224)):
        self.model = model
        self.im_size = im_size

        self.biases = []
        self.feature_grads = []
        self.grad_handles = []

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                
                # Register feature-gradient hooks for each layer
                handle_g = m.register_backward_hook(self._extract_layer_grads)
                self.grad_handles.append(handle_g)

                # Collect model biases
                b = self._extract_layer_bias(m)
                if (b is not None): self.biases.append(b)


    def _extract_layer_bias(self, module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different 
        # from batchnorm bias parameter. 
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b) 
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm2d):
            b = - (module.running_mean * module.weight 
                    / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    def getBiases(self):
        # dummy function to get biases
        return self.biases

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            self.feature_grads.append(out_grad[0])

    def getFeatureGrads(self, x, output_scalar):
        
        # Empty feature grads list 
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs = output_scalar, inputs = x)[0]

        return input_gradients, self.feature_grads

def compute_saliency(dataloader,image_size):
    maps = []
    device = torch.device('cpu')
    model = CNNSingleValueRanker(image_size)
    model.load_state_dict(torch.load('RankPrediction-model.pkl'))
    model = model.to(device)
    unnormalize = NormalizeInverse(mean = [0, 0, 0],std = [1,1,1])
    for batch_idx, (data, _) in enumerate(dataloader):
        
        data = data.to(device).requires_grad_()
        # Compute saliency maps for the input data
        fullgrad_method = FullGrad(model,im_size= (3,image_size,image_size))
        saliency_map = fullgrad_method.saliency(data)

        # Get saliency maps
        for i in range(data.size(0)):
            image = data[i].cpu()
            image = unnormalize(image)
            single_map = get_saliency_map(image, saliency_map[i])
            maps.append(single_map)
    return maps