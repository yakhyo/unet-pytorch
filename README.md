## Implementation of [UNet](https://arxiv.org/abs/1505.04597) in PyTorch

<div align="center">
<img src="assets/image.jpg" width="45%">
<img src="assets/mask.gif" width="45%">
</div>

## Getting Started

### Dataset

[Carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data) dataset is used to train and test
the model.
To download the dataset run the `download.sh` file.

**Note:** Please download `kaggle.json` file first from associated kaggle account then save it under the `/home/username/.kaggle/kaggle.json` (in linux)

### Train

```commandline
git clone git@github.com:yakhyo/UNet-pt.git
cd UNet-pt
python3 train.py
```

Epoch Loss Graph after 5 epochs (Not mean, sum):

<div align="center">
<img src="assets/loss_graph.png" width="70%">
</div>

### Inference

```commandline
python inference.py --input [input_image_path] --output [output_image_path]
```
