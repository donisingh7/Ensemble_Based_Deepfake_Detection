# Ensemble-Based Deepfake Detection

Research project exploring **deepfake image classification** with multiple CNN backbones and ensemble strategies.

The repository contains experiments with individual transfer-learning models and two ensemble approaches:

- **Soft voting** across multiple CNN predictions
- **Stacking** with classical/meta models over CNN outputs

## Models explored

The repository includes experiments built around:

- VGG19
- EfficientNet
- ResNet50
- MobileNetV2
- Xception
- ensemble meta-models using Logistic Regression, Random Forest, XGBoost, and a small neural network

## Repository structure

| Path | Purpose |
| --- | --- |
| `first step(any approach from relevant research paper)/` | Initial baseline experiments |
| `Second Step (create model with some other approach)/` | Alternate deep-learning approach |
| `Third step/` | EfficientNet experiment |
| `Fourth Step ResNet50 Model/` | ResNet50 experiment |
| `Fifth Step MobileNetV2 Model/` | MobileNetV2 experiment |
| `Ensemble_Learning_Soft_Voting/` | Soft-voting + stacking ensemble experiments |
| `DeepFake_Detection_Research_Paper_Minor_Project.pdf` | Research paper artifact |
| `Mini_Project_DeepFake_Detection_Project_Report.pdf` | Project report |

## Ensemble architecture

```text
Input image
   │
   ├── VGG19
   ├── EfficientNet
   ├── ResNet50
   ├── MobileNetV2
   └── Xception
        │
        ▼
Predicted probabilities
        │
        ├── Soft voting
        │
        └── Stacking meta-models
             ├── Logistic Regression
             ├── Random Forest
             ├── XGBoost
             └── Neural Network
                    │
                    ▼
              Final prediction
```

## Results captured in the notebooks

This repository preserves the original experimental outputs rather than rewriting them into a cleaner-looking benchmark.

From `Ensemble_Learning_Soft_Voting/1_Soft_voting.ipynb`:

- Soft-voting ensemble accuracy: **0.4702**

From `Ensemble_Learning_Soft_Voting/2. Stacking_Ensemble.ipynb`:

- XGBoost meta-model accuracy: **0.6498**
- XGBoost ROC-AUC: **0.7182**

These numbers should be interpreted as the results of the exact stored experiments, not as a general benchmark claim. Individual backbone notebooks also show substantial variation and signs of generalization difficulty, which was one reason to explore ensemble approaches.

## What this project demonstrates

- transfer learning with multiple CNN backbones
- binary image classification
- confidence/probability fusion
- soft-voting ensembles
- stacking ensembles
- classical ML meta-models on deep-model predictions
- evaluation using accuracy, classification reports, confusion matrices, ROC curves, and ROC-AUC
- research iteration where weaker approaches are retained for comparison instead of hidden

## Reproducing the work

The notebooks were developed experimentally and are preserved as research artifacts. To reproduce them:

1. Create a Python environment.
2. Install the dependencies in `requirements.txt`.
3. Prepare the dataset expected by the notebooks.
4. Update local dataset/model paths where required.
5. Run the individual backbone notebooks before running the ensemble notebooks.

Model checkpoint files are retained in the repository because they are part of the original project artifact. For future work, large trained models should preferably be published through GitHub Releases, object storage, or Git LFS rather than committed directly.

## Limitations

- Some notebooks contain local-path assumptions that may require adjustment before rerunning.
- The stored experiments are exploratory and do not represent a production deepfake detector.
- The soft-voting result is weaker than the later stacking experiment.
- A stronger follow-up would use a clean train/validation/test protocol, unified preprocessing, reproducible configuration, and model calibration.

## Author

**Doni Singh Agrawal**

Research project in deep learning, computer vision, and ensemble learning.
