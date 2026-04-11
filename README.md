""" sample usage (train)
# classification (always train this first)
python train.py --task classification --epochs 20 --lr 1e-4 --dropout-p 0.3 --use-wandb

# localization (warm-start encoder from classifier)
python train.py --task localization --epochs 15 --lr 1e-4 \
    --encoder-checkpoint checkpoints/classifier.pth --use-wandb

# segmentation — full fine-tuning (best strategy per 2.3)
python train.py --task segmentation --epochs 15 --lr 5e-5 \
    --encoder-checkpoint checkpoints/classifier.pth --freeze-strategy none --use-wandb

# segmentation — frozen encoder (for 2.3 ablation)
python train.py --task segmentation --epochs 15 --lr 1e-4 \
    --encoder-checkpoint checkpoints/classifier.pth --freeze-strategy encoder --use-wandb

# segmentation — partial fine-tuning
python train.py --task segmentation --epochs 15 --lr 5e-5 \
    --encoder-checkpoint checkpoints/classifier.pth --freeze-strategy partial --use-wandb
"""

""" sample usage (inference)
# single-task evaluation on val/test
python inference.py --task classification --split val
python inference.py --task localization --split val
python inference.py --task segmentation --split val

# multitask (uses all three checkpoints from checkpoints/)
python inference.py --task multitask --split test


# override a specific checkpoint
python inference.py --task classification --split test \
    --checkpoint-path checkpoints/classifier_best.pth
"""