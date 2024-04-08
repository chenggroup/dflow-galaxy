# DFlow Galaxy

[![PyPI version](https://badge.fury.io/py/dflow-galaxy.svg)](https://badge.fury.io/py/dflow-galaxy)
[![Downloads](https://pepy.tech/badge/dflow-galaxy)](https://pepy.tech/project/dflow-galaxy)
[![Docker](https://img.shields.io/docker/v/link89/dflow-galaxy?label=docker&logo=docker)](https://hub.docker.com/repository/docker/link89/dflow-galaxy/general)


Collection of workflows and tools built on top of DFlow and [ai2-kit](https://github.com/chenggroup/ai2-kit).

## Features
* DFlowBuilder: A type friendly wrapper for `DFlow` to build workflows in a more intuitive way.
* Workflows:
  * TESLA: a **T**raining-**E**xploration-**S**creening-**L**abeling workflow designed by **AI4EC**. Get inspired by [DPGEN](https://github.com/deepmodeling/dpgen), [DPGEN2](https://github.com/deepmodeling/dpgen2), ported from [ai2-kit](https://github.com/chenggroup/ai2-kit).


## Get Started
`dflow-galaxy` requires Python 3.9+ since it depends on `typing.Annotated`.

To use `dflow-galaxy` as a library, you can install it via pip:

```bash
pip install dflow-galaxy
```

For developers, please use `poetry` to bootstrap the development environment:

```bash
poetry install
poetry shell
```

## Bohrium Launch App
This repository also contains some launch apps for Bohrium. The entry of the Bohrium launch app is in the `launch_app` directory, and relative modules are in the `dflow_galaxy/app` directory.

Currently, the following launch apps are available:

* CP2K Lightning: Run CP2K without building input files from scratch.
* DynaCat TESLA: An automated workflow to generated DeepMD potential for DynaCat. (TODO)
* DynaCat MD: Run MD simulations with DeepMD potential for DynaCat. 


## Distribution
### PyPI
```bash
poetry publish --build
```
### Docker
```bash
./build-docker.sh
```

## Manuals
* [TESLA Workflow](doc/tesla.md)
