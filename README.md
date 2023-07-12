# UAV Billboards - dataset, detection and tracking 

![header](./README_FILES/header.webp)

## Description

We propose a deep-learning-based system for the automatic detection of billboards using consumer-grade unmanned aerial vehicles (UAVs). Thanks to the geospatial information from the device and sensor data, the position of billboards can be estimated.

## Dataset

The dataset contains 1361 images supplemented with additional spatial metadata, together with 5210 annotations in a COCO-like format. It can be downloaded from [here](https://chmura.put.poznan.pl/s/lIMsy8OlOjuXAIJ).

| **Object name** 	| **Count** 	| **Example** 	|
|:---------------:	|:---------:	|:-----------:	|
|  free-standing  	|    3694   	| <img src="./README_FILES/example_0.png" width="200px" height="200px"> |
|   wall-mounted  	|    1284   	| <img src="./README_FILES/example_1.png" width="200px" height="200px"> |
| large road sign 	|    232    	| <img src="./README_FILES/example_8.png" width="200px" height="200px"> |


## Usage

### Training scripts

All training scripts and pipelines are located in the `./ml` directory. Next instructions are placed in [this file](./ml/README.md).

### Application

The application is located in the `./app` directory. Tnstallation steps and guide are placed in [this file](./app/README.md).
