# DA6401 Assignment 2- Building a Complete Visual Perception Pipeline  

**Name:** Tanmay Gawande
**Roll Number:** DA25M030

## Links

- **GitHub Repository:** [Public repository link](https://github.com/Tanmay20030516/DA6401-assignment-2)
- **W&B Report:** [Public report link](https://wandb.ai/da25m030-tanmay-gawande/da6401_assignment2/reports/DA6401-Assignment-2-Building-a-Complete-Visual-Perception-Pipeline--VmlldzoxNjQ5MDI1Nw?accessToken=vcmtidura4foj40yxhj9ed555avouwaa3xlhv18ads9x2iabduexyb3onsireh0g)


## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   └── iou_loss.py
├── models/
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   ├── multitask.py
│   ├── layers.py
│   └── vgg11.py
├── train.py
├── inference.py
└── requirements.txt
```


## Training

Train the classification model first, then use its encoder checkpoint to warm-start the other tasks.

```bash
# 1. Classification
python train.py --task classification --epochs 20 --lr 1e-4 --dropout-p 0.3 --use-wandb

# 2. Segmentation (full fine-tuning from classifier encoder)
python train.py --task segmentation --epochs 15 --lr 5e-5 \
    --encoder-checkpoint checkpoints/classifier.pth --freeze-strategy none --use-wandb
```

## Inference

```bash
# Evaluate classification on validation split
python inference.py --task classification --split val

# Evaluate multitask model on test split
python inference.py --task multitask --split test
```