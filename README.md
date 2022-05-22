### Dataset Distillation with per-point Auto Augmentation
Postech CSED499I-02 source code and sample

### Environments:
- matplotlib(3.5.1)
- numpy(1.21.5)
- python(3.8.13)
- pytorch(1.11.0)
- pyyaml(6.0)
- scikit-learn(1.0.2)
- torchvision(0.12.0)

### Key Implementation Details:
- [Augmentation Module](https://github.com/rach-rgb/DDAA/blob/main/src/custom_augment/augmentation.py)
- [Distillation Module](https://github.com/rach-rgb/DDAA/blob/main/src/distillation.py) 
- [Loss Model](https://github.com/rach-rgb/DDAA/blob/main/src/loss_model.py)

### Directories:
- Configuration: `./configs/*.yaml`
- Auto-Augmentation Search Configuration: `./configs/search/*.yaml`
- Dataset: `./data/MNIST/` or `./data/cifar-10-*`
- Output:
  - Distilled dataset: `./output/result.pth`
  - Auto-Augmentation Policy Module: `./output/project_weights.pth` and `./output/extractor_weights.pth`
  - Intermediate Visualized Dataset: `./output/epoch*/`
  - Final Visualized Dataset: `./output/visuals_step*.png`
  - logs: `./output/logging.log` or `./output/*-search-log.txt`

### Train & Evaluation Demos
- Auto-Augmentation Explore:
`python search.py explore-CIFAR10.yaml`
- Dataset Distillation:
`python main.py dd-auto-CIFAR10.yaml`
- Evaluate Distilled Dataset
`python main.py test-dd-CIFAR10.yaml`
- Store Augmented Dataset
`python augment_dataset.py augment-CIFAR10.yaml`


### Results
- Dataset Distillation w/ Auto-Augmentation
- Dataset Distillation w/ Class-balanced Loss  
![Cross Entropy, Class Balanced Focal Loss, Class Balanced Cross Entropy를 사용하여 Class Imbalance한 Dataset에 Distillation을 적용했을 때 생성한 Distilled Dataset의 품질을 평가한 그래프이다.](https://github.com/rach-rgb/DDAA/blob/main/table.png "Dataset Distillation w/ Class Balanced Loss for CIFAR-10")

