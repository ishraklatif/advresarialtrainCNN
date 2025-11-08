Vision Transformer (ViT) Adversarial Training for Object Classification

This repository implements object classification with adversarial robustness using a Vision Transformer (ViT) backbone from timm
.
It supports transfer learning, PGD adversarial attack training, and advanced data augmentation techniques (MixUp and CutMix).

🚀 Features

Transfer Learning – Fine-tune pretrained ViT backbones from timm.

Adversarial Training – Integrated PGD attack for robust model learning.

Augmentation Strategies – MixUp and CutMix implemented from scratch.

Configurable Architecture – Easily swap models, tweak optimizer, or change hyperparameters.

Checkpoint Management – Save, resume, and tag best validation or robust checkpoints.

Inference & Submission Pipeline – Generates labeled predictions with optional Test-Time Augmentation (TTA).

🧩 Repository Structure
.
├── attacks.py         # PGD adversarial attack implementation
├── augment.py         # MixUp and CutMix augmentations
├── checkpoints.py     # Save/load checkpoints and state dicts
├── config.py          # Global configs (paths, hyperparams, seed)
├── data.py            # Data loading, preprocessing, and TTA utilities
├── engine.py          # Trainer class handling train/eval/robust accuracy
├── inference.py       # Test-time prediction + submission CSV
├── main.py            # Entry point for training & inference
├── model_utils.py     # ViT creation, layer freezing, parameter groups
├── utils.py           # Helpers: seeding, cosine LR schedule, normalization
└── README.md

⚙️ Installation
git clone https://github.com/ishraklatif/advresarialtrainCNN.git
cd <repo-name>
pip install -r requirements.txt


Key dependencies:

torch torchvision timm numpy pandas matplotlib tensorboard

🏋️‍♂️ Training
1. Configure

Update config.py:

MODEL_NAME = "vit_base_patch16_224"
EPOCHS = 30
BATCH_SIZE = 32
LR = 5e-5
EPS = 8/255
ALPHA = 2/255
PGD_STEPS = 5
USE_TTA = True

2. Run training
python main.py --train


During training:

MixUp and CutMix are applied for regularization.

PGD adversarial examples are generated for robust optimization.

Model checkpoints (best_val, best_robust) are saved automatically.

🧪 Evaluation
Validation
python main.py --eval

Robust Accuracy under PGD

Automatically computed via:

Trainer.evaluate_pgd(model, val_loader)

🔮 Inference

To generate predictions for the test set:

python main.py --inference


Outputs a CSV file submission.csv containing:

ID,Label
1,Cat
2,Car
...


Supports Test-Time Augmentation (--tta) for improved stability.

🧠 Adding PEFT (Optional)

You can optionally integrate LoRA or adapter-based PEFT to fine-tune only a small subset of parameters for faster, memory-efficient training.
See the lora_vit.py
 template example in the documentation section.

📊 Monitoring

Training metrics are logged using TensorBoard:

tensorboard --logdir runs/

🧱 Checkpoints

Each run saves:

best_val.ckpt – Best validation accuracy.

best_robust.pth – Best robust (adversarial) accuracy.

last.ckpt – Last epoch snapshot.

🔍 Example: PGD Attack Function
x_adv = x.clone().detach()
for _ in range(steps):
    x_adv.requires_grad_()
    loss = F.cross_entropy(model((x_adv - mean_t)/std_t), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = (x_adv + alpha * grad.sign()).clamp(0., 1.)
    delta = torch.clamp(x_adv - x, -eps, eps)
    x_adv = (x + delta).clamp(0., 1.)

📈 Results
Metric	Baseline	+Adversarial	+MixUp/CutMix	+PEFT (LoRA)
Val Acc	87.3%	85.2%	89.1%	88.8%
Robust Acc	51.4%	69.7%	72.0%	80.5%

(Example metrics; replace with your experiment results.)

🧰 Future Improvements

Add LoRA/adapter-based PEFT for efficient fine-tuning.

Integrate Free Adversarial Training for faster convergence.

Support multi-GPU distributed training.

Add grad-CAM visualization for interpretability.

📜 Citation

If you use this repository, please cite:

@software{vit_adv_training_2025,
  author = {Ishrak Latif},
  title  = {Adversarially Robust Vision Transformer},
  year   = {2025},
  url    = {https://github.com/<your-username>/<repo-name>}
}