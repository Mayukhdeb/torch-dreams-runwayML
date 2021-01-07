# runway_model.py
import numpy as np
from PIL import Image
from torch_dreams.dreamer import dreamer
from torch_dreams.utils import preprocess_numpy_img

import torchvision.models as models
import torchvision.transforms as transforms
import torch
import cv2

# from model_utils import model, generate_mask, segmentation_model, layers

import runway
from runway.data_types import image,  number, boolean

"""
works on torch_dreams v1.1.0
for tests use: $ pip install git+https://github.com/Mayukhdeb/torch-dreams
to run server on localhost: $ python runway_model.py
"""

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


config = {
    "image_path": None,
    "layers": [model.inception4b],  
    "custom_func": [None]
}

input_dict = { 
    "image": image(), 
    "octave_scale": number(step = 0.05, min = 1.0, max = 1.7, default = 1.2), 
    "num_octaves":number(step = 1, min = 1, max = 25, default = 5),
    "iterations" : number(step = 1, min = 1, max = 100, default = 14),
    "lr": number(step = 1e-4, min = 1e-9, max = 1e-1, default = 0.05),
    "max_rotation": number(step = 0.1, min = 0.0, max = 1.5, default = 0.9)
}

for key in list(layers.keys()):
    input_dict[key] = boolean(default = False)


@runway.setup
def setup():
    dreamy_boi = dreamer(model)
    return  dreamy_boi

@runway.command(
    name = "generate", 
    inputs= input_dict, 
    outputs={ "image": image() }
)
def generate(dreamy_boi, input):

    image_np = preprocess_numpy_img(np.array(input["image"]).astype(np.float32)/255.0)

    """
    generate mask
    """
    mask = generate_mask(model = segmentation_model, image = np.array(input["image"]), invert= False, factor = 2.0)   

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

    for key in list(layers.keys()):
        if input[key] == True and layers[key] not in config["layers"]:
            config["layers"].append(layers[key])

        elif input[key] == False and layers[key] in config["layers"]:
            if layers[key] in config["layers"]:
                config['layers'].remove(layers[key])
    # print(len(config["layers"]))

    out = dreamy_boi.deep_dream_with_masks(config)*255
    out = Image.fromarray(out.astype(np.uint8))
    return { "image": out }

"""
after running this, open runwayML and connect to localhost
"""
if __name__ == "__main__":
     runway.run()