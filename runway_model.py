# runway_model.py
import json
from PIL import Image
import numpy as np
import torchvision.models as models
from torch_dreams.dreamer import dreamer

import runway
from runway.data_types import image, text, number
"""
to run server on localhost
python runway_model.py
"""

model = models.inception_v3(pretrained=True)

config = {
    "image_path": None,
    "layers": [model.Mixed_5c.branch3x3dbl_3],  ## change this 
    "octave_scale": 1.1,
    "num_octaves": 14,
    "iterations": 10,
    "lr": 0.03,
    "max_rotation": 0.5,
}

@runway.setup
def setup():

    dreamy_boi = dreamer(model)
    return  dreamy_boi

@runway.command(
    name = "generate", 
    inputs={ 
        "image_path": text(), 
        "octave_scale": number(step = 0.01, min = 1.0, max = 1.7), 
        "num_octaves":number(step = 1, min = 1, max = 25),
        "iterations" : number(step = 1, min = 1, max = 500),
        "lr": number(step = 1e-4, min = 1e-9, max = 1e-2),
        "max_rotation": number(step = 0.01, min = 0.0, max = 1.5)
        }, 
    outputs={ "image": image() }
)
def generate(dreamy_boi, input):

    config["image_path"] = input["image_path"]
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
then enter text input: "images/sample_small.jpg"
"""
if __name__ == "__main__":
     runway.run()