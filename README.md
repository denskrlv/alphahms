# AlphaHMS

<p align="center">
  <img src="hero.jpg" alt="AlphaHMS hero image" width="100%" />
  <br/>
  <sub>Photo by <a href="https://unsplash.com/@theshubhamdhage?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Shubham Dhage</a> on <a href="https://unsplash.com/photos/a-brain-displayed-with-glowing-blue-lines-2sz-3NrmZYU?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></sub>
</p>

**⚡ Multi-modal Graph Neural Networks for Harmful Brain Activity Classification**

AlphaHMS is a deep-learning pipeline for classifying harmful brain activity from EEG and spectrogram recordings, built around the [HMS Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) Kaggle challenge. The system represents each recording as a pair of temporal graph sequences and learns a joint EEG + spectrogram representation with Graph Attention Networks, BiLSTM temporal encoders, hierarchical regional pooling, and cross-modal attention fusion.

🎯 The six target classes are: **Seizure**, **LPD** (Lateralized Periodic Discharges), **GPD** (Generalized Periodic Discharges), **LRDA** (Lateralized Rhythmic Delta Activity), **GRDA** (Generalized Rhythmic Delta Activity), and **Other**.

---

## ✨ Highlights

- 🕸️ **Graph-based EEG modelling** — each 50 s EEG recording is split into 9 overlapping 10 s windows; nodes are the 19 EEG channels and edges are derived from inter-channel coherence (threshold 0.5).
- 🌈 **Graph-based spectrogram modelling** — 600 s spectrograms are split into 119 windows over 4 spatial regions (LL, RL, LP, RP) with fixed spatial connectivity.
- 🧩 **Hierarchical pooling** by clinical brain regions (Frontal, Central, Parietal, Occipital) before temporal modelling.
- 🔀 **Cross-modal fusion** with multi-head attention between EEG and spectrogram regional embeddings.
- ⚡ **PyTorch Lightning** training with mixed precision (BF16), WandB logging, cross-validation, class-weighted / KL-divergence losses, and early stopping.
- 🔍 **Explainability** via GNNExplainer and attention-weight inspection.
- 📊 **Baselines** included: EEG-only GNN and a raw-EEG MLP.

---

## 📁 Repository Layout

```
AlphaHMS/
├── configs/                       # OmegaConf YAML configs
│   ├── graphs.yaml                # Preprocessing parameters
│   ├── model.yaml                 # Multi-modal GNN architecture
│   ├── model_eeg.yaml             # EEG-only baseline architecture
│   ├── train.yaml                 # Main training config
│   ├── train_4fold.yaml           # Cross-validation training
│   ├── train_eeg.yaml             # EEG-only baseline training
│   ├── training_mlp.yaml          # MLP baseline training
│   ├── inference_mlp.yaml         # MLP inference
│   └── smoke_test.yaml            # Quick smoke test
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis & preprocessing
├── src/
│   ├── data/                      # Datasets, DataModules, graph builders
│   │   ├── graph_dataset.py
│   │   ├── graph_datamodule.py
│   │   ├── baseline_dataset.py
│   │   ├── baseline_datamodule.py
│   │   ├── raw_eeg_dataset.py
│   │   ├── raw_datamodule.py
│   │   ├── make_graph_dataset.py  # Build graph dataset from raw data
│   │   └── utils/
│   │       ├── eeg_process.py
│   │       └── spectrogram_process.py
│   ├── models/
│   │   ├── hms_model.py           # Multi-modal model
│   │   ├── hms_eeg_model.py       # EEG-only baseline
│   │   ├── eeg_mlp.py             # MLP baseline
│   │   ├── regularization.py
│   │   ├── explainer_wrappers.py
│   │   └── graph_layers/          # GAT, temporal, fusion, pooling, classifier
│   ├── lightning_trainer/         # LightningModules (multi-modal, EEG, MLP)
│   ├── explainers/
│   │   └── gnn_explainer.py
│   ├── train.py                   # Training entrypoint (GNN models)
│   ├── train_mlp.py               # Training entrypoint (MLP baseline)
│   ├── explain.py                 # GNNExplainer driver
│   ├── explain_attention.py       # Attention-weight analysis
│   └── explain_model.py
├── tests/                         # Pytest suite
├── inspect_data.py                # Quick data inspection utility
├── environment.yaml               # Conda environment specification
└── pytest.ini
```

---

## 🛠️ Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yaml -y
conda activate graph
```

### 2. Install PyTorch with the correct CUDA wheel

The environment file deliberately omits PyTorch so you can match your local CUDA version. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torcheeg
```

You will also need **PyTorch Geometric** matching your PyTorch / CUDA build — follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## 📦 Dataset

### 1. Download the raw HMS data

You must accept the competition terms on Kaggle first.

```bash
mkdir -p data/raw && cd data/raw
kaggle competitions download -c hms-harmful-brain-activity-classification
unzip hms-harmful-brain-activity-classification.zip
```

Expected layout under `data/raw/`:

```
data/raw/
├── train.csv
├── train_eegs/            # parquet files, one per EEG recording
└── train_spectrograms/    # parquet files, one per spectrogram
```

### 2. Run the EDA / preprocessing notebook

```bash
jupyter execute notebooks/eda.ipynb
```

### 3. Build the graph dataset

```bash
python src/data/make_graph_dataset.py
```

This produces one `data/processed/patient_{id}.pt` per patient plus a `metadata.pt` index. See [src/data/README.md](src/data/README.md) for full details on graph construction, output format, and memory requirements.

---

## 🚀 Training

All training scripts log to **Weights & Biases**; run `wandb login` once before starting.

### Multi-modal GNN (main model)

```bash
python src/train.py --train-config configs/train.yaml
```

### Cross-validation (5-fold)

```bash
python src/train.py --train-config configs/train_4fold.yaml
```

### EEG-only GNN baseline

```bash
python src/train.py --train-config configs/train_eeg.yaml
```

### MLP baseline (raw EEG)

```bash
python src/train_mlp.py --config configs/training_mlp.yaml
```

### Quick smoke test

```bash
python src/train.py --train-config configs/smoke_test.yaml
```

Evaluation runs automatically at the end of training. Checkpoints are written to the directory specified in the config.

---

## 🏗️ Model Architecture

The multi-modal model (see [configs/model.yaml](configs/model.yaml)) is composed of:

1. **EEG encoder** — 2-layer multi-head GAT (64-dim, 4 heads) with coherence edge weights → hierarchical regional pooling → 2-layer BiLSTM (128-dim, bidirectional).
2. **Spectrogram encoder** — 2-layer GAT (64-dim, 4 heads) over the 4 spatial regions → BiLSTM (128-dim, bidirectional).
3. **Cross-modal fusion** — multi-head cross-attention (8 heads, 256-dim) between EEG and spectrogram regional embeddings, with attention pooling over regions.
4. **Classifier** — MLP with hidden sizes `[256, 128]`, ELU activations, dropout 0.3 → 6-class softmax.

Loss defaults to **KL divergence** against the soft expert-vote distribution; class weighting, graph-Laplacian regularization, and edge-weight penalties are all configurable.

---

## 🔬 Explainability

```bash
# GNNExplainer over a trained checkpoint
python src/explain.py

# Attention-weight visualisation
python src/explain_attention.py
```

---

## 🧪 Testing

```bash
pytest
```

The test suite covers the data module, preprocessing pipeline, multiprocessing, regularization, checkpoint resume, and spectrogram processing.

---

## 💻 Hardware Notes

- Training was developed on H200 / RTX-class GPUs with **BF16 mixed precision**.
- Preprocessing is CPU-bound and benefits from `build_workers` set to your physical core count.
- Recommended: ≥ 16 GB RAM for preprocessing, ≥ 1 modern CUDA GPU for training.

---

## 📎 Citation

If you use AlphaHMS in your research or build upon it, please cite this repository:

```bibtex
@software{krylov2025alphahms,
  author       = {Denis Krylov, Samuel Goldie, Alberto Pasinato, Serkan Akin, Leonardo Lago},
  title        = {{AlphaHMS}: Multi-modal Graph Neural Networks for Harmful Brain Activity Classification},
  year         = {2025},
  institution  = {Delft University of Technology},
  url          = {https://github.com/deniskrylov/AlphaHMS},
  note         = {Built for the HMS Harmful Brain Activity Classification challenge (Kaggle)}
}
```
