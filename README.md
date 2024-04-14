<h1 align="center">Welcome to MiLMo 🤖⛏️💎⛏💎🪨⛏</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://twitter.com/ArnolFokam" target="_blank">
    <img alt="Twitter: ArnolFokam" src="https://img.shields.io/twitter/follow/ArnolFokam.svg?style=social" />
  </a>
</p>

> This project demonstrates how to train a small GPT model on Minecraft maps encoded as 3D NumPy arrays. The 
repository also includes scripts for obtaining or generating data from Minecraft maps while the game is running.

## Quickstart

### Pre-requisites

- Python >= `3.8`
- Minecraft Launcher >= `1.12.2`
- Java 8 aka `1.8`

The following steps assumed the repository is already cloned and you are on a terminal with a working python environment.

### Install Project Requirements

- Install PyTorch from the steps outline [here](https://pytorch.org/get-started/locally/).
- Install project requirements from the specified file.

```bash
pip install -r requirements.txt
```

### Run the Minecraft server and client

More information can be found [here](https://github.com/real-itu/Evocraft-py#4-rendering-minecraft).

## Usage

### Save a Minecraft map as Training Data

- Download a minecraft map at [https://www.minecraftmaps.com/](https://www.minecraftmaps.com/). Make sure that it is compatible with the Minecraft version `1.12.2` (see available maps [here](https://www.minecraftmaps.com/1-12-2)).

- Extract the downloaded zip file and copy the **map folder** in the repository.
- The map folder should have a structure similar to the following structure:

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

- Update the server configuration in the file `server.properties` by setting `level-name=NAME OF COPIED MAP FOLDER`


## Author

👤 **Arnol Fokam**

* Website: https://arnolfokam.github.io/
* Twitter: [@ArnolFokam](https://twitter.com//ArnolFokam)
* Github: [@ArnolFokam](https://github.com/ArnolFokam)
* LinkedIn: [@arnolfokam](https://linkedin.com/in/arnolfokam)

## Show your support

Give a ⭐️ if this project helped you or you just like what is being built!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_