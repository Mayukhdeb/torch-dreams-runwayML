# torch-dreams :handshake: runwayML
<div style="text-align:center">
<img src = "images/vis.png" width = "50%">
</div>

## How to run

0. Clone the repo and navigate into the folder.
    ```
    $ git clone https://github.com/Mayukhdeb/torch-dreams-runwayML.git
    $ cd torch-dreams-runwayML
    ```

1. Start the server with: 
    ```
    $ python runway_model.py
    ```

    Or if you want to use the segmentation mask, use:

    ```
    $ python runway_model_with_segmentation.py
    ```

You should see an output similar to this:
```
Initializing model...
dreamer init on:  cuda
Model initialized (0.04s)
Starting model server at http://0.0.0.0:9000...
```

2. Open a new workspace in [RunwayML](https://learn.runwayml.com/#/getting-started/installation) and press `connect`. Make sure your localhost port matches the one given by the server script (in our case it's `9000`)

3. Select the image you want to work with.

4. Play around with the inference parameters and have fun! 

Try reading the [torch-dreams docs](https://app.gitbook.com/@mayukh09/s/torch-dreams/) to create fancier visualizations. 

## Architecture
Each Runway model consists of two special files:

- [`runway_model.py`](runway_model.py): A Python script that imports the runway module (SDK) and exposes its interface via one or more `@runway.command()` functions. This file is used as the **entrypoint** to your model.
- [`runway.yml`](runway.yml): A configuration file that describes dependencies and build steps needed to build and run the model. 

### The `runway_model.py` Entrypoint File

The [`runway_model.py`](runway_model.py) entrypoint file is the file the Runway app will use to query the model. This file can have any name you want, but we recommend calling it `runway_model.py`.


### The `runway.yml` Config File

Each Runway model must have a [`runway.yml`](runway.yml) configuration file in its root directory. This file defines the steps needed to build and run your model for use with Runway. This file is written in YAML, a human-readable superset of JSON. Below is an example of a `runway.yml` file. This example file illustrates how you can provision your model’s environment.

```yaml
version: 0.1
python: 3.6
entrypoint: python runway_model.py
cuda: 10.2
framework: pytorch
files:
    ignore:
        - image_dataset/*
build_steps:
    - pip install runway-python==0.1.0
    - pip install -r requirements.txt
```
