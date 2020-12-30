# runway_model.py
import json
from PIL import Image
import numpy as np
import torchvision.models as models
from torch_dreams.dreamer import dreamer
from torch_dreams.utils import preprocess_numpy_img

import runway
from runway.data_types import image, text, number

"""
works on torch_dreams v1.0.1
for tests use: $ pip install git+https://github.com/Mayukhdeb/torch-dreams
to run server on localhost: $ python runway_model.py
"""

model = models.inception_v3(pretrained=True)

config = {
    "layers": [model.Mixed_6c.branch1x1],  ## change this 
}

@runway.setup
def setup():

    dreamy_boi = dreamer(model)
    return  dreamy_boi

@runway.command(
    name = "generate", 
    inputs={ 
        "image": image(), 
        "octave_scale": number(step = 0.01, min = 1.0, max = 1.7, default = 1.3), 
        "num_octaves":number(step = 1, min = 1, max = 25, default = 5),
        "iterations" : number(step = 1, min = 1, max = 500, default = 14),
        "lr": number(step = 1e-4, min = 1e-9, max = 1e-1, default = 0.05),
        "max_rotation": number(step = 0.1, min = 0.0, max = 1.5, default = 0.9)
        }, 
    outputs={ "image": image() }
)
def generate(dreamy_boi, input):

    config["image"] = preprocess_numpy_img(np.array(input["image"]).astype(np.float32)/255.0)
    config["octave_scale"] = input["octave_scale"]
    config["num_octaves"] = input["num_octaves"]
    config["iterations"] = input["iterations"]
    config["lr"] = input["lr"]
    config["max_rotation"] = input["max_rotation"]
    out = dreamy_boi.deep_dream(config)*255
    out = Image.fromarray(out.astype(np.uint8))
    return { "image": out }

"""
after running this, open runwayML and connect to localhost
"""
if __name__ == "__main__":
     runway.run()