# runway_model.py
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import cv2


"""
set up segmentation utils
"""
segmentation_model = models.segmentation.fcn_resnet101(pretrained=True).eval()
model = models.googlenet(pretrained=True).eval()


seg_transforms = transforms.Compose([                 
                            transforms.ToPILImage(),
                            transforms.Resize(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                        std = [0.229, 0.224, 0.225])])

def generate_mask(model, image, transforms = seg_transforms, invert = False, factor = 1.0):
    inp = seg_transforms(image).unsqueeze(0)
    out = model(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    om = np.array([om,om,om]).transpose(1,2,0).astype(np.float32)
    out= cv2.resize(om, (image.shape[1], image.shape[0]))
    out = out/out.max()

    if invert is True:
        out = 1-out

    out *= factor
    return out


layers = {
    "inception3a": model.inception3a,
    "inception3b": model.inception3b,
    "inception4a":model.inception4a,
    "inception4b":model.inception4b,
    "inception4c":model.inception4c,
    "inception4d":model.inception4d,
    "inception4e":model.inception4e,
    "inception5a":model.inception5a,
    "inception5b":model.inception5b,
}