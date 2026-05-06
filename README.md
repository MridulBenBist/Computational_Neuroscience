# CNN vs. Transformer Representations of Animacy and Memorability

A computational neuroscience project comparing how a supervised CNN (**ResNet-50**) and a vision-language transformer (**CLIP ViT-B/16**) represent two properties that matter for the human ventral visual stream: **animate vs. inanimate category structure** and **image memorability**.

The goal is not to crown a winner, but to show how unit-level selectivity and response magnitude differ between architectures and training objectives, and whether either model's internal signals track human behavior.

---

## What's in this repo

Two analyses, each as a self-contained Colab/Jupyter notebook:

### 1. Animacy selectivity on MemCat (`CompNeuroProject_lastEXP.ipynb`)
- Computes a **d′-based selectivity index** (pooled within-group SD on raw activations) for every final-layer unit in ResNet-50 and CLIP ViT-B/16.
- Compares two CLIP feature heads: **CLS token** (canonical) and **patch-mean** (symmetric to ResNet's avg-pool).
- Visualizes the top-9 activating images and spatial activation maps for the most animate- and inanimate-selective units.
- Tests whether top-selective units' activations correlate with human memorability **within** each animacy category (Spearman ρ with vectorized bootstrap CIs).
- Animacy mapping: `animate = {animal}`, `inanimate = {food, landscape, vehicle}`. The `sports` category is excluded because it conflates animate subjects with inanimate context.

### 2. Response magnitude vs. memorability on LaMem (`FinalVersion.ipynb`)
- Extracts response magnitudes for ~10,000 LaMem images:
  - **ResNet-50:** L2 norm of spatially-averaged `layer4` activations.
  - **CLIP ViT-B/16:** divisive normalization — CLS norm / mean spatial-patch norm.
- Wilcoxon rank-sum tests for high vs. low memorability splits, plus Spearman ρ between magnitude and memorability score.
- **GradCAM** (ResNet) and **attention rollout** (CLIP) overlays for the highest-magnitude high- and low-memorability images.
- Rank-averaged CNN+Transformer ensemble.

---

## Datasets

- **MemCat** — 10,000 images with human memorability scores across 5 categories (animal, food, landscape, vehicle, sports). Expected at `MyDrive/MemCat_data/` with the metadata CSV `memcat_image_data.csv`.
- **LaMem** — 58,741 images with memorability scores; auto-downloaded from `chitradrishti/LAMEM` on Hugging Face.

---

## Setup

A CUDA-capable GPU is required (T4 or better is fine). Both notebooks were developed in Google Colab.

```bash
pip install timm open_clip_torch torch torchvision pandas scipy \
            scikit-learn matplotlib seaborn pillow tqdm \
            grad-cam huggingface_hub transformers
```

---

## How to run

### MemCat animacy notebook
1. Mount Google Drive and place MemCat at `MyDrive/MemCat_data/MemCat/` with `memcat_image_data.csv` alongside it.
2. Set runtime to **T4 GPU**, then **Runtime → Restart runtime → Run all**.
3. Outputs land in `/content/memcat_proj/`:
   - `figures/fig1_selectivity_dist.png` — d′ distributions across models
   - `figures/fig2_top_images.png` — top-9 activators for most selective units
   - `figures/fig3_cam_overlay.png` — spatial activation overlays
   - `figures/fig4_memorability_modulation.png` — within-category Spearman ρ with bootstrap CIs
   - `summary.json` — quantitative summary

### LaMem magnitude notebook
1. Run cells in order — the dataset auto-downloads and extracts (~one-time, a few GB).
2. Outputs in the working directory:
   - `fig1_wilcoxon.png` — magnitude × memorability scatter + boxplots
   - `fig2_gradcam_rollout.png` — GradCAM vs. attention rollout panel
   - `fig3_model_comparison.png` — head-to-head and rank-averaged ensemble

---

## Methodological notes

- **Selectivity** uses Cohen's d on raw activations with pooled within-group SD. Earlier global z-scoring biased d′ toward zero by absorbing the between-group difference into the scale.
- **CLIP features** are reported with both CLS token and patch-mean heads so conclusions can't pivot on a single arbitrary choice. A Spearman ρ between the two heads' per-unit selectivity vectors is reported as an agreement check.
- **All correlations are Spearman.** Magnitudes (L2 norms, ratios) are skewed and non-Gaussian; the memorability score is bounded. Rank-based statistics are appropriate.
- **Ensemble** combines models by rank-averaging rather than z-scoring, so heavy tails in either magnitude distribution can't dominate.
- **Bootstrap CIs** (1000 resamples) use a vectorized rank-Pearson trick (Pearson on ranks ≡ Spearman) — roughly 50× faster than per-iteration scipy calls.

---

## File layout

```
.
├── CompNeuroProject_lastEXP.ipynb   # MemCat animacy selectivity
├── FinalVersion.ipynb               # LaMem magnitude × memorability
└── README.md
```

---

## Caveats

- Drive I/O is the bottleneck for MemCat; if extraction is slow, copy images to local disk first or drop `NUM_WORKERS` to 2.
- CLIP "spatial activation" maps are unit activity on the patch grid, **not** saliency — interpret accordingly.
- Memorability scores reflect human group behavior, not individual recall, and the LaMem/MemCat datasets have known content biases.

---

