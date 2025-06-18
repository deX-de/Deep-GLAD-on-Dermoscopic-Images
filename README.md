# From Pixels to Graphs: Deep Graph-Level Anomaly Detection on Dermoscopic Images

This is the source code for our systematic comparison of image-to-graph transformations and graph-level anomaly detection methods mentioned in our paper "Deep Graph-Level Anomaly Detection on Dermoscopic Images". We implement a graph-based approach to (one vs. rest) anomaly detection on the HAM10000 image datasets using Graph Neural Networks (GNNs). 
The code currently supports and implements the following models: 
[OCGTL](https://github.com/boschresearch/GraphLevel-AnomalyDetection), [SIGNET](https://github.com/yixinliu233/SIGNET), and [CVTGAD](https://github.com/jindongli-Ai/CVTGAD).

## Features

- Implements various Graph-level Anomaly Detection models
- Configurable graph construction methods (SLIC / Patch segmentation)
- Various node feature options (mean RGB, color, texture, shape, or all combined)
- Edge construction methods (K-Nearest Neighbors and Region Adjacency Graph)

## Requirements

- Python == 3.9
- PyTorch == 2.0.1
- PyTorch Geometric == 2.6.1
- NumPy == 1.22.4
- Pandas
- PyYAML
- tqdm
- scikit-learn == 1.0.2
- scikit-image == 0.19.3
- OpenCV (for image processing)

## Installation

- Clone this repository:
```
git clone https://github.com/deX-de/Deep-GraphLevel-Anomaly-Detection-on-Dermoscopic-Images.git
```
- Create a new [conda](https://repo.anaconda.com/archive) environment:
```
conda create -n gad python=3.9
conda activate gad
```
- Install the required packages:
```
pip install -r requirements_cuda_1.txt
pip install -r requirements_cuda_2.txt
```
or

```
pip install -r requirements_cpu_1.txt
pip install -r requirements_cpu_2.txt
```

## Usage

### Dataset Creation

To create and inspect a dataset, use the `create_graph_dataset.py` script:
```
mkdir data
python create_graph_dataset.py <path/to/config>
```
This script will create or load the specified dataset and display information about it.

### Running Experiments

To run the set of experiments, use the `run_experiments.py` script (modes: `[unsupervised, semi_supervised]`):

```
mkdir results
mkdir logs
python run_experiments.py --mode <mode> --configs <path/to/config1> <path/to/config2> ...
```

This script will:
- Run experiments for all specified datasets, models, and configurations
- Save individual experiment results to a CSV file
- Generate a summary of results

## Configuration
To recreate results on HAM10000, download the following files from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T):

- HAM10000_images_part_1.zip
- HAM10000_images_part_2.zip
- HAM10000_metadata.tab (as Comma Separated Values)
- HAM10000_segmentations_lesion_tschandl.zip

Then:
- Unzip both HAM10000_images_part_1.zip and HAM10000_images_part_2.zip and move content into the directory data/HAM10000/raw/images
- Unzip the content of HAM10000_segmentations_lesion_tschandl to data/HAM10000/raw/masks 
- Move HAM10000_metadata.csv to data/HAM10000/raw


## License

This project is licensed under the AGPL-3.0 license. See [LICENSE](LICENSE) for details.
