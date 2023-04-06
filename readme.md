# An-Empirical-Study-of-Overlooked-Code-Reviews

This repository contains the main data and scripts used in 'An Empirical Study of Overlooked Code Reviews'.


## Dataset

* analyse.xlsx

  * It contains the result of 10 projects analysed in this study (Section 3.2).
  * We recorded the no-feedback comment in the 'comment' field and its category in the 'TYPE' field.
  * We also recorded the author and lasting time of the pull requests in the 'author' and 'lasting time' fields.
  * You can find the analyzed pull requests through the 'PROJECT' and 'PR ID' fields.

* data_of_ten_projects

  * The folder contains relevant data of ten projects crawled in this article, including pull requests, commits, and comments.

* training_dataset

  * The folder contains the data used for training.
  * label = 0 means a comment received feedback.
  * label = 1 means a comment did not receive feedback.
        
* result

  * The folder contains the results obtained after training.
        
* main.py

  * It contains the code for our model training, that is, the implementation of classification techniques.

### Installation


## Results

The following is the stratified 10-fold average performance of different classifiers mentioned in the papaer.

|     Model      | Pos-Precision | Pos-Recall |   Pos-F1   | Neg-Precision | Neg-Recall |   Neg-F1   |  Accuracy  |
| :------------: | :-----------: | :--------: | :--------: | :-----------: | :--------: | :--------: | :--------: |
| `comment only` |    77.35%     |   60.70%   |   67.29%   |    93.14%     |   96.49%   |   94.77%   |   90.99%   |
|    `concat`    |    76.11%     |   62.53%   |   67.93%   |    93.42%     |   96.20%   |   94.77%   |   91.02%   |
|  `concat+MLP`  |    78.16%     |   57.89%   |   66.14%   |    92.71%     | **97.00%** |   94.80%   |   90.99%   |
|  `MLP+concat`  |    77.58%     |   62.15%   |   68.57%   |    93.39%     |   96.71%   |   95.01%   |   91.39%   |
|   `gatting`    |  **80.21%**   | **63.95%** | **70.38%** |  **93.71%**   |   96.96%   | **95.29%** | **91.89%** |


## Usage

### Installation

Create a `Python==3.7` environment using `conda` or `miniconda`, if you don't have a conda environment on your device, you can download it from [here](https://docs.conda.io/en/main/miniconda.html).
```shell
git clone https://github.com/no-feedback-comment-db/Overlooked-Code-Reviews.git
cd Overlooked-Code-Reviews

conda create --name py37 python=3.7 -y
conda activate py37
pip3 install -r requirements.txt
```

### Reproduce

* To reproduce `gatting` model mentioned in the paper, use the following command:

```shell
python3 main.py --task "gatting_10fold" \
                --data_root "./training_dataset" \
                --result_root "./results"\
                --model "gatting" \
                --global_seed 413 \
                --train_epochs 3 \
                --learning_rate 1e-5 \
                --over_write
```
* To reproduce `comment only` model mentioned in the paper, use the following command:

```shell
python3 main.py --task "comment_10fold" \
                --data_root "./training_dataset" \
                --result_root "./results"\
                --model "comment only" \
                --global_seed 413 \
                --train_epochs 3 \
                --learning_rate 1e-5 \
                --over_write
```

* To reproduce `concat` model mentioned in the paper, use the following command:

```shell
python3 main.py --task "concat_10fold" \
                --data_root "./training_dataset" \
                --result_root "./results"\
                --model "concat" \
                --global_seed 413 \
                --train_epochs 3 \
                --learning_rate 1e-5 \
                --over_write
```

* To reproduce `concat+mlp` model mentioned in the paper, use the following command:

```shell
python3 main.py --task "concat+mlp_10fold" \
                --data_root "./training_dataset" \
                --result_root "./results"\
                --model "concat+mlp" \
                --global_seed 413 \
                --train_epochs 3 \
                --learning_rate 1e-5 \
                --over_write
```

* To reproduce `mlp+concat` model mentioned in the paper, use the following command:

```shell
python3 main.py --task "mlp+concat_10fold" \
                --data_root "./training_dataset" \
                --result_root "./results"\
                --model "mlp+concat" \
                --global_seed 413 \
                --train_epochs 3 \
                --learning_rate 1e-5 \
                --over_write
```

