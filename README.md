<h1 align="center">Welcome to MiLMo: <b>Mi</b>necraft <b>L</b>anguage <b>Mo</b>del 🤖⛏️💎⛏💎🪨⛏</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

[**Overview**](#overview)
| [**Quickstart**](#quickstart)
| [**Usage**](#usage)
| [**Contributing**](#contributing)
| [**Author's Info**](#author)

<h2 name="overview" id="overview">Overview 🔍</h2>

MiLMo is a project that demonstrates the training of a small GPT model on Minecraft maps encoded as 3D NumPy arrays. The repository also includes scripts for obtaining or generating data from Minecraft maps while the Minecraft server is running.

<h2 name="quickstart" id="quickstart">Quickstart 🚀</h2>

### System Requirements

- Python >= `3.8`
- Minecraft Launcher >= `1.12.2`
- Java 8 aka `1.8`

The following steps assume the repository is already cloned and you are on a terminal with a working Python environment.

### Install Project Requirements

- Install PyTorch from the steps outlined [here](https://pytorch.org/get-started/locally/).
- Install project requirements from the specified file.

```bash
pip install -r requirements.txt
```

### Run the Minecraft server and client

More information can be found [here](https://github.com/real-itu/Evocraft-py#4-rendering-minecraft).

<h2 name="usage" id="usage">Usage 🪛</h2>

### Save a Minecraft map as Training Data

- Download a Minecraft map at [https://www.minecraftmaps.com/](https://www.minecraftmaps.com/). Ensure that it is compatible with Minecraft version `1.12.2` (see available maps [here](https://www.minecraftmaps.com/1-12-2)).

- Extract the downloaded zip file and copy the **map folder** into the repository.
- The map folder should have a structure similar to the following:

```bash
├── advancements
├── data 
├── DIM1 
├── DIM-1 
├── playerdata
├── region
├── stats
├── datapacks
└── level.dat
```

- Update the server configuration in the file [server.properties](/server.properties) by setting `level-name` to the name of the folder containing the map.

- Run the server with the following command:

```bash
java -jar spongevanilla-1.12.2-7.3.0.jar
```

This command runs the Minecraft server with the downloaded map.

- Use the script [create_dataset.py](/create_dataset.py) to create the dataset. This script will:
    - read the map from the running server
    - convert a portion of the map into a `numpy.ndarray` and save it.

- Running this script will save the extracted portion of the world at [data/worlds/](/data/worlds/) as an `npy` file containing a 3D volume of integers.

### Train a GPT model on a saved map

- Set `data_dir` in [exps/base.yaml](/exps/base.yaml) to the path of the saved map.
- Ensure that `experiment.do_pretraining` is set to `True` while `experiment.do_generation` is set to `False`.
- Then, run the following command:

```bash
python main.py --config-path=exps --config-name=gpt
```

This command will train and save the model at `results/XXXX-XX-XX/XX-XX-XX/model.pth`.

### Generate Minecraft data with trained model

- To generate maps with a training model at `results/XXXX-XX-XX/XX-XX-XX/model.pth`, you should:
  - Set `experiment.do_generation` to `True` and `experiment.do_pretraining` to `False`.
  - Set `generation.pretrained_model_path` to the path to the trained model.
  - Run the following command:

  ```bash
  python main.py --config-path=exps --config-name=gpt
  ```

This command will generate two folders in `results/XXXX-XX-XX/XX-XX-XX/` named `generations` and `samples` where the former contains `npy` files of generations made by the model with some start tokens identical to `npy` files in `samples` which contains the actual true values.

### Visualize a generated map

- Update the server configuration in the file [server.properties](/server.properties) by setting `level-name` to `world`.

- Run the server with the following command:

```bash
java -jar spongevanilla-1.12.2-7.3.0.jar
```

This command runs the Minecraft server on an empty map onto which we will place our generated blocks.

- When the server is running, run the following:

```bash
python viz.py --saved_blocks_dir <PATH_TO_GENERATED_NPY_FILE>
```

Replace `<PATH_TO_GENERATED_NPY_FILE>` with the actual path to the generated `.npy` file.

<h2 name="contributing" id="contributing">Contributing 🤝</h2>

I do not accept contributions for now, but feel free to raise any issues you spot with the source code.

<h2 name="author" id="author">Author's Info 👨‍🎨</h2>

* Website: https://arnolfokam.github.io/
* Twitter: [@ArnolFokam](https://twitter.com/ArnolFokam)
* Github: [@ArnolFokam](https://github.com/ArnolFokam)
* LinkedIn: [@arnolfokam](https://linkedin.com/in/arnolfokam)