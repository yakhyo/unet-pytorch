# [UNet](https://arxiv.org/abs/1505.04597) Implementation using PyTorch | Car Segmentation

<table>
  <tr>
    <td style="text-align: center;">
      <p>Input image</p>
      <img src="assets/image.jpg" alt="Input image" width="250">
    </td>
    <td style="text-align: center;">
      <p>Predicted mask</p>
      <img src="assets/image_mask.png" alt="Predicted mask" width="250">
    </td>
    <td style="text-align: center;">
      <p>Image and mask overlay</p>
      <img src="assets/image_overlay.png" alt="Image and mask overlay" width="250">
    </td>
  </tr>
</table>

## Getting Started

```
git clone git@github.com:yakhyo/unet-pytorch.git
cd unet-pytorch
```

- [x] Dice loss and Cross Entropy loss used for training. See the [dice loss](unet/utils/loss.py) implementation. `dice_score = 1 - dice_loss` used for evaluation.
- [x] Model weight provided in `weights` folder. Weights saved in f16 (~60MB).
- [x] [Demo Jupyter Notebook](demo.ipynb).

To Do:
- [] ONNX export.
- [] ONNX inference.

### Dataset

[Carvana Image Masking (PNG)](https://www.kaggle.com/datasets/ipythonx/carvana-image-masking-png) dataset is used to train the model. After downloading the data place them under `./data` directory.

```
├── data
    ├── train_images
         ├── xxx.jpg
         ├── xxy.jpg
         ├── xxz.jpg
          ....
    ├── train_masks
         ├── xxx.png
         ├── xxy.png
         ├── xxz.png
```

### Training

Training arguments

```
usage: train.py [-h] [--data DATA] [--scale SCALE] [--num-classes NUM_CLASSES] [--weights WEIGHTS] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--num-workers N] [--lr LR] [--weight-decay WEIGHT_DECAY] [--momentum MOMENTUM] [--amp] [--print-freq PRINT_FREQ]
                [--resume RESUME] [--use-deterministic-algorithms] [--save-dir SAVE_DIR]

UNet training arguments

options:
  -h, --help            show this help message and exit
  --data DATA           Directory containing the dataset (default: './data')
  --scale SCALE         Scale factor for input image size (default: 0.5)
  --num-classes NUM_CLASSES
                        Number of output classes (default: 2)
  --weights WEIGHTS     Path to pretrained model weights (default: '')
  --epochs EPOCHS       Number of training epochs (default: 10)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 4)
  --num-workers N       Number of data loading workers (default: 8)
  --lr LR               Learning rate (default: 1e-5)
  --weight-decay WEIGHT_DECAY
                        Weight decay (default: 1e-8)
  --momentum MOMENTUM   Momentum (default: 0.9)
  --amp                 Enable mixed precision training
  --print-freq PRINT_FREQ
                        Frequency of printing training progress (default: 10)
  --resume RESUME       Path to checkpoint to resume training from (default: '')
  --use-deterministic-algorithms
                        Forces the use of deterministic algorithms only.
  --save-dir SAVE_DIR   Directory to save model weights (default: 'weights')
```

Train the model

```commandline
python train.py
```

### Inference

Inference arguments

```
usage: inference.py [-h] [--model-path MODEL_PATH] [--image-path IMAGE_PATH] [--scale SCALE] [--save-overlay]

Image Segmentation Inference

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to the model weights
  --image-path IMAGE_PATH
                        Path to the input image
  --scale SCALE         Scale factor for resizing the image
  --save-overlay        Save the overlay image if this flag is set
```

Inference

```
python inference.py --model-path weights/last.pt --image-path assets/image.jpg
```
