# Arabic Poetry Mood Labeling (with Resilient Checkpointing)

An advanced, mathematically calibrated classification pipeline that labels contiguous Arabic poetry verse-batches along four independent semantic axes using BGE-M3 text embeddings and Z-score standardization.

This repository features a robust, chunk-based checkpointing framework designed specifically for long-running Google Colab sessions, ensuring you never lose your progress during expensive embedding generation.

---

## 🌟 Key Features

1. **Four-Axis Taxonomy**: Labels poetry batches along independent dimensions:
   - **Mood**: فرح (Joyful), حزن (Sorrowful), غضب (Defiant), تشاؤم (Bleak), تفاؤل وأمل (Hopeful), حنين وشوق (Nostalgic), وحدة (Isolated), شكوى (Complaining), عتاب (Reproachful), تأمل (Contemplative).
   - **Genre**: مدح (Praise), هجاء (Satire), رثاء (Elegy), غزل (Romantic), فخر (Boastful), حكمة (Wise/Philosophical), زهد (Ascetic/Devotional), وصف (Descriptive), خمريات (Revelrous), حماسة (Heroic/Martial).
   - **Energy**: هادئ جدا (Calm/Ambient), هادئ (Mellow/Low Energy), متوسط (Moderate), نشيط (Upbeat/Energetic), شديد الحماس (Intense/High-Energy).
   - **Aesthetic**: تراثي أصيل (Traditional/Oud), ملحمي أوركسترالي (Epic/Orchestral), صوفي روحاني (Mystical/Spiritual), عسكري حماسي (Martial/Drums), رومانسي عاطفي (Tender/Ballad), حزين كئيب (Melancholic/Somber), احتفالي شعبي (Festive/Folk).

2. **Hub-Label Bias Elimination via Z-Scores**: 
   Standard vector similarity search against prompts is often biased towards specific generic "hub prompts." On-the-fly **Z-Score Normalization** standardizes label cosine similarities dynamically across the entire dataset, creating precise and reliable class boundaries.

3. **Fault-Tolerant Chunked Checkpoint System**:
   - **Atomic Writes**: Writes each chunk to a temporary file first and renames it atomically (`os.replace`) to guarantee no corrupted checkpoints are created in a crash mid-write.
   - **Incremental Git Sync**: Commit and push embeddings to GitHub in the background after every `N` chunks automatically. If Google Colab times out, disconnects, or crashes, **re-running the script picks up exactly where it left off**.
   - **Safe Recovery**: If git operations fail due to a flaked connection, the script keeps running and writing locally, queueing the commits for the next successful check.

---

## 🛠️ File Inventory

- `label_moods.py`: The core pipeline. Orchestrates loading the dataframe, running chunked embeddings, Z-Score calibration, tag assignment, confidence calculation, and output exporting.
- `checkpoint_utils.py`: Manages the atomic loading/saving of NumPy chunk files, configuration alignment validation, and best-effort Git synchronization.
- `arabic_poetry_moode_labeling.ipynb`: Interactive Jupyter Notebook designed to easily run the entire pipeline in Google Colab with Google Drive mounts and secure GitHub Token parameters.

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have the required packages:
```bash
pip install pandas sentence-transformers numpy tqdm
```

### 2. Basic Command Line Usage

Run the labeling script by passing the path to the input pickle dataframe:
```bash
python label_moods.py TOP_100_ARABIC_POETS_OF_ALL_TIME_STAGE_02.pkl \
  --device cuda \
  --batch-size 128 \
  --checkpoint-dir checkpoints \
  --chunk-size 500 \
  --push-every 4
```

### 3. CLI Arguments

- `pkl_path`: Path to input pkl dataframe (required).
- `--out`: Output pkl path (default: `[input_stem]_mood_labeled.pkl`).
- `--device`: Target device (`cuda` or `cpu`).
- `--batch-size`: Encoder batch size (default: `64`).
- `--checkpoint-dir`: Directory for saving chunked `.npy` checkpoints (default: `checkpoints`).
- `--chunk-size`: Size of row-order chunks for checkpointing (default: `500`).
- `--push-every`: Git commit and push checkpoints every `N` completed chunks (default: `4`).
- `--no-git-push`: Save checkpoints locally on disk but disable remote GitHub syncing.
- `--repo-dir`: Path to the Git repository root (default: `.`).

---

## 💡 Architecture Detail: Why Chunk Only Embeddings?

Generating embeddings with deep Transformer models on ~24,000 rows represents 99.9% of the computational cost of this pipeline. Conversely, cosine similarity matrix multiplication and Z-score calibration run in a matter of seconds.

Therefore, our architecture separates the pipeline cleanly into **two phases**:
1. **Stage A (Expensive / Resumable)**: Embed text in row-order chunks, saving completed `.npy` files to disk atomically.
2. **Stage B (Fast / Deterministic)**: Concatenate the saved embedding chunks, calculate taxonomy matrices, perform Z-score standardization on the global population distribution, and export the labeled dataset. 

If a run fails during Stage B, re-running is virtually free because Stage A reads the pre-computed embedding chunks directly from disk.

---

## 📊 Standardizing Equations

Given a matrix of raw cosine similarities $S \in \mathbb{R}^{N \times L}$ for $N$ verses and $L$ labels on an axis, the standardized Z-Score $Z_{i,j}$ for verse $i$ and label $j$ is calculated as:

$$Z_{i,j} = \frac{S_{i,j} - \mu_j}{\sigma_j}$$

Where:
- $\mu_j$ is the column-wise mean similarity of label $j$ across all $N$ verses.
- $\sigma_j$ is the column-wise standard deviation of similarity of label $j$ across all $N$ verses.
- If $\sigma_j < 10^{-9}$, it is set to $1.0$ to prevent division by zero.
