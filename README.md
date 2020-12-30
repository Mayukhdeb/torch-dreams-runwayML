# torch-dreams :handshake: runwayML

## How to run

1. Start the server with: `$ python runway_model.py`. You should see an output similar to this:
```
Initializing model...
dreamer init on:  cuda
Model initialized (0.04s)
Starting model server at http://0.0.0.0:9000...
```

2. Then open a new workspace in [RunwayML](https://learn.runwayml.com/#/getting-started/installation) and press `connect`. Make sure your localhost port matches the one given by the server script (in our case it's `9000`)

3. Enter the filename of the image (relative to the script).

4. Play around with the inference parameters and have fun!

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
cuda: 9.2
framework: tensorflow
files:
    ignore:
        - image_dataset/*
build_steps:
    - pip install runway-python==0.1.0
    - pip install -r requirements.txt
```

### Testing your Model

While you're developing your model it's useful to run and test it locally.

```bash
## Optionally create and activate a Python 3 virtual environment
# virtualenv -p python3 venv && source venv/bin/activate

# Install the Runway Model SDK (`pip install runway-python`) and the Pillow
# image library, used in this example.
pip install -r requirements.txt

# Run the entrypoint script
python runway_model.py
```

You should see an output similar to this, indicating your model is running.

```
Setting up model...
[SETUP] Ran with options: seed = 0, truncation = 10
Starting model server at http://0.0.0.0:8000...
```

You can test your model once its running by POSTing a caption argument to the the `/generate` command.

```bash
curl \
    -H "content-type: application/json" \
    -d '{ "caption": "red" }' \
    http://localhost:8000/generate
```

You should receive a JSON object back, containing a cryptic base64 encoded URI string that represents a red image:

```
{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."}
```
