# Image Classification with ViT-Tiny: Comparing CE, ArcFace, and CAC Losses

In this project, the **ViT-tiny** backbone is trained using various loss functions to perform a 172-class image classification task. The performance and representations are compared across Precision, Recall, and F1 metrics, as well as embedding space quality evaluated through kNN and t-SNE analysis.

## Models Compared

- **SimCLR+head (Cross-Entropy):** The backbone was pre-trained self-supervised with SimCLR, then a classifier head was added for fine-tuning.
- **CE (Cross-Entropy):** Standard supervised baseline.
- **ArcFace (Additive Angular Margin):** $s=30, m=0.5$, Embedding dimension = 128.
- **CAC (Class Anchor Clustering):** $\alpha=10, \lambda=0.1$, Embedding dimension = number of classes (172), with fixed anchors.

> **Note:** The difference between SimCLR+head and CE lies solely in the pre-training strategy, whereas the difference between ArcFace and CAC stems from the choice of loss function. Avoid confounding these two factors when interpreting the results.

---

## Dataset

- **Train:** 35,587 images, 172 classes
- **Val:** 8,893 images, 172 classes

The dataset is not included in the repository due to its large size. The directory structure is expected as follows:

```text
split_output/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

> **Important:** The training and validation sets must use an identical class-to-index mapping; otherwise, evaluation metrics will be invalid.

---

## Training Setup

### ArcFace & CAC Settings
- **Backbone:** `vit_tiny_patch16_224` via `timm`
- **Optimizer:** AdamW ($\text{lr} = 3\times10^{-4}$, $\text{weight\_decay} = 0.05$) with Cosine Annealing scheduler
- **Precision:** Mixed 16-bit (`16-mixed`)
- **Random Seed:** `42`
- **Epochs:** 10 (Best checkpoint selected based on highest `val_f1_macro`)
- **Framework:** PyTorch Lightning

### CE & SimCLR+Head Settings
- *[Insert your epoch, learning rate, batch size, and SimCLR pre-training configurations here]*

---

## Evaluation Methodology

1. **Native Predictions:**
   - **CE:** $\text{argmax}(\text{logits})$
   - **ArcFace:** Highest cosine similarity
   - **CAC:** Nearest class anchor
2. **kNN (Embedding Quality):**
   - Training embeddings act as the gallery, and validation images act as queries ($k=20$, cosine distance, temperature $\tau = 0.07$). This method is classifier-head-agnostic, providing a uniform benchmark for all representations.
3. **Metrics:**
   - Accuracy, Macro and Weighted Precision / Recall / F1-score.
   - Macro F1 includes a **95% Bootstrap Confidence Interval**.

---

## Results

### Training Dynamics (Validation Set)
After 10 epochs, both ArcFace and CAC reached approximately **0.76 Macro F1** and **0.84 Accuracy** on the validation set.
- **CAC** learns faster during the initial epochs (reaching ~0.52 Macro F1 in Epoch 1 compared to ~0.43 for ArcFace).
- By the final epoch, this gap closes completely. Evaluating one loss as strictly "superior" based on a single seed over 10 epochs is inconclusive.

### Embedding Quality (kNN, Validation Set)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **SimCLR+head** | **0.6845** | **0.6223** | **0.5330** | **0.5518** |
| **CAC** | 0.6613 | 0.5900 | 0.5077 | 0.5214 |
| **ArcFace** | 0.6523 | 0.5680 | 0.4768 | 0.4895 |
| **CE** | 0.5192 | 0.4644 | 0.3566 | 0.3705 |

**Key Takeaway:** Standard CE embeddings perform significantly worse compared to models trained with margin-based or anchor-based losses. The SimCLR pre-trained model yielded the highest kNN performance overall.

### Native Predictions (Validation Set)
*This table will be populated after verifying the evaluation script's label mapping. Data to be populated from `report/final_comparison.csv` and `report/summary.md`.*

### Embedding Space Visualization (t-SNE)
- t-SNE was fitted on 3,000 randomly sampled validation images.
- Due to color palette repetition over 172 classes, the visualization serves for qualitative assessment only.
- **Observations:** Standard CE displays large, overlapping clusters, whereas SimCLR+head and ArcFace form noticeably tighter, well-separated clusters.

---

## Repository Structure

```text
.
├── notebooks/               # Training notebooks for CE, ArcFace, CAC, and SimCLR
├── evaluate_models_v2.py    # Script for evaluating models and generating reports
├── report/                  # Outputs: CSVs, plots, and summary.md
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Set up the environment:**
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # Linux/macOS:
   # source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Configure evaluation parameters:**
   Update the parameters at the top of `evaluate_models_v2.py`:
   - `TRAIN_DIR`, `VAL_DIR`: Dataset paths
   - `CKPT`: Paths to the four model checkpoints
   - `IMG_SIZE`, `MEAN`, `STD`: Ensure preprocessing matches training setups

3. **Execute evaluation:**
   ```bash
   python evaluate_models_v2.py
   ```

Output files will be saved in the `report/` directory: `final_comparison.csv`, `per_class_f1.csv`, `top_confusions_<model>.csv`, `metrics_bar.png`, `training_curves.png`, `tsne.png`, and `summary.md`.

* **Model Weights (`.pth`, `.ckpt`):** Excluded from the repository due to file size constraints. [Download Weights Link]

---

## Limitations & Considerations

- **Single Seed & Short Schedule:** Each model was trained with a single random seed for 10 epochs. Minor differences may be due to stochastic variance; if bootstrap confidence intervals overlap, the difference should not be considered statistically significant.
- **Validation Selection Bias:** The best checkpoint was chosen based on validation score, making validation metrics slightly optimistic. Testing on an isolated test set is recommended.
- **Hyperparameter Tuning:** Hyperparameters for ArcFace ($s, m$) and CAC ($\alpha, \lambda$) were not systematically tuned.
- **Experimental Consistency:** If SimCLR+head and CE setups differ from ArcFace/CAC in training configurations, performance gaps cannot be attributed purely to the loss functions.