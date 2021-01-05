# runway_model.py
import json
from PIL import Image
import numpy as np
import torchvision.models as models
from torch_dreams.dreamer import dreamer
from torch_dreams.utils import preprocess_numpy_img
import torchvision.transforms as transforms
import torch
import cv2

import runway
from runway.data_types import image, text, number

"""
works on torch_dreams v1.0.1
for tests use: $ pip install git+https://github.com/Mayukhdeb/torch-dreams
to run server on localhost: $ python runway_model.py
"""

"""
set up segmentation utils
"""
segmentation_model = models.segmentation.fcn_resnet101(pretrained=True).eval()
seg_transforms = transforms.Compose([                 
                            transforms.ToPILImage(),
                            transforms.Resize(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                        std = [0.229, 0.224, 0.225])])

def generate_mask(model, image, transforms = seg_transforms):
    inp = seg_transforms(image).unsqueeze(0)
    out = model(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    om = np.array([om,om,om]).transpose(1,2,0).astype(np.float32)
    out= cv2.resize(om, (image.shape[1], image.shape[0]))
    return out

model = models.googlenet(pretrained=True)

def custom_func(layer_outputs):
    loss = layer_outputs[0][7].mean() 
    return loss

config = {
    "layers": [model.inception4d],  ## change this 
    "custom_func": [custom_func]
}


@runway.setup
def setup():

    dreamy_boi = dreamer(model)
    return  dreamy_boi

@runway.command(
    name = "generate", 
    inputs={ 
        "image": image(), 
        "octave_scale": number(step = 0.05, min = 1.0, max = 1.7, default = 1.3), 
        "num_octaves":number(step = 1, min = 1, max = 25, default = 5),
        "iterations" : number(step = 1, min = 1, max = 100, default = 14),
        "lr": number(step = 1e-4, min = 1e-9, max = 1e-1, default = 0.05),
        "max_rotation": number(step = 0.1, min = 0.0, max = 1.5, default = 0.9)
        }, 
    outputs={ "image": image() }
)
def generate(dreamy_boi, input):

    image_np = preprocess_numpy_img(np.array(input["image"]).astype(np.float32)/255.0)

    """
    generate mask
    """
    mask = generate_mask(model = segmentation_model, image = np.array(input["image"]))   

    """
    generate output with grad mask 
    """
    config["image"] = image_np
    config["octave_scale"] = input["octave_scale"]
    config["num_octaves"] = input["num_octaves"]
    config["iterations"] = input["iterations"]
    config["lr"] = input["lr"]
    config["max_rotation"] = input["max_rotation"]
    config["grad_mask"] = [mask]

    out = dreamy_boi.deep_dream_with_masks(config)*255
    out = Image.fromarray(out.astype(np.uint8))
    return { "image": out }

"""
after running this, open runwayML and connect to localhost
"""
if __name__ == "__main__":
     runway.run()