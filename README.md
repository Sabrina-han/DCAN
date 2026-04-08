# DMSP — Dataset for Multimodal Personality Research

Code for **DMSP: A Multimodal Dataset for MBTI Personality and Fairness Analysis**

## 💡 Introduction

The **DMSP** is a pioneering resource designed to address the challenges in personality detection from multimodal content, particularly focusing on the **Myers-Briggs Type Indicator (MBTI)** and **fairness evaluation**. Unlike existing datasets that rely heavily on text-only inputs or binary labels, our dataset is carefully curated to integrate **Visual, Audio, and Textual modalities**. By incorporating fairness attributes (Gender, Age, Race) and continuous soft labels, this dataset offers a more accurate reflection of how personality traits manifest in real-world scenarios.

**Key Features:**
* **Multimodal Integration**: Leverages CLIP (Visual), Wav2Clip (Audio), and CLIP Sentence Embeddings (Text) for robust feature representation.
* **Fairness-Oriented**: Includes demographic annotations (Gender, Age, Race) to facilitate fairness analysis and bias mitigation in AI models.
* **Soft Labeling**: Utilizes continuous scores for the 4 MBTI dimensions (E/I, N/S, F/T, J/P), moving beyond the limitations of hard binary classifications.
* **Open for Research**: Designed to advance work in both affective computing and algorithmic fairness.

---

## 📚 Dataset
📢 Availability Note: We provide partial dataset samples in the  directory for reference. The complete dataset (839 samples in total) will be publicly released upon acceptance of the paper.
The **DMSP** dataset originally comprised **839 samples**. The dataset is located in the `DMSP/` directory.

### Directory Structure

```text
DMSP/
├── train.csv                 # Training labels and metadata
├── test.csv                  # Test labels and metadata
├── train_clipimage.pkl       # Visual features for training set (CLIP ViT-B/32)
├── test_clipimage.pkl        # Visual features for test set
├── train_audio_wav2clip.pkl  # Audio features for training set (Wav2Clip)
├── test_audio_wav2clip.pkl   # Audio features for test set
├── train_clipsentence.pkl    # Text features for training set (CLIP Sentence)
└── test_clipsentence.pkl     # Text features for test set
```

### Label Description

Each sample in `train.csv` and `test.csv` contains soft scores for MBTI dimensions alongside demographic metadata:

| Field | Description | Type | Statistics (Train) |
| :--- | :--- | :--- | :--- |
| `VideoName` | Unique video identifier | `string` | - |
| `score_[DIM]` | Soft score (0–1) for MBTI dimensions (E/I, N/S, F/T, J/P) | `float` | Mean ~0.5-0.6 |
| `Gender` | Demographic label (1=Male, 2=Female) | `int` | 1 (~54%), 2 (~46%) |
| `Age` | Age group label | `int` | 15-17 (Peak at 16) |

---

## 🔧 Data Construction & Annotation

### Soft Label Generation

To mitigate the subjectivity inherent in personality assessment, we adopt the **Soft Label Construction** methodology proposed by [MbtiBench](https://github.com/Personality-NLP/MbtiBench). Instead of relying on hard binary classifications, we generate continuous probability distributions using the **Expectation-Maximization (EM) Algorithm**.

* **Raw Annotation**: Each sample is annotated by multiple experts using **fine-grained intensity labels** (e.g., `E+` for strong Extraversion, `I-` for weak Introversion).
* **EM Aggregation**: The EM algorithm estimates the ground-truth probability distribution from noisy, subjective annotations by weighing annotator consistency and label intensity.
* **Result**: Final labels are **continuous soft scores (0–1)**. For example, an `E/I` score of `0.80` provides significantly more information about the subject's tendency toward Introversion than a binary label alone.

### DMSP Extensions

Building upon MbtiBench, DMSP extends the dataset in two critical dimensions:

1. **Demographic Annotation**: Gender, Age, and Race attributes annotated for fairness evaluation.
2. **Multimodal Data Integration**: Soft labels grounded in rich video data across three modalities:
   * **Visual**: Facial expressions and gestures (CLIP ViT-B/32, 768-dim, 15 sampled frames)
   * **Audio**: Tonal and prosodic features (Wav2Clip, 512-dim, 15 audio segments)
   * **Text**: Verbal content (CLIP Sentence Embeddings, 768-dim)

---

## 🏗️ Usage & Evaluation

### ⚙️ Environment Setup

```bash
pip install -r requirements.txt
```

### 💻 Dataset Loading

```python
from train_FMPD_MBTI_baseline_fixed import DMSPDataset

train_ds = DMSPDataset(
    csv_file='DMSP/train.csv',
    data_dir='DMSP',
    split='train'
)

sample = train_ds[0]
print(sample.keys())
# Output: ['vid', 'mbti', 'demo', 'visual', 'audio', 'text']
```

### 🚀 Model Training

```bash
python train_FMPD_MBTI_baseline_fixed.py \
      --data_dir DMSP \
      --use_bacl \
      --use_facl
```

### ⚖️ Fairness Evaluation

```bash
python calculate_mbti_fairness.py --data_dir DMSP
```

---

## 📊 Comparison with MbtiBench

| Aspect | MbtiBench | DMSP |
| :---: | :---: | :---: |
| **Modalities** | Text-only | **Multimodal (Text, Audio, Visual)** |
| **Label Type** | Hard (Binary) / Soft | **Soft (Continuous)** |
| **Fairness** | None | **Gender, Age** |
| **Features** | Sentence Embeddings | **CLIP, Wav2Clip** |

---
## Ethical Statement

Informed consent was obtained from all student participants for the collection and use of identifiable video data (including clear facial footage) for academic research purposes. Participation was voluntary, and data will be used strictly for analysis and dissemination in anonymized forms within research outputs.


## 🧭 Acknowledgements

We thank the developers of **MbtiBench** for the original benchmark design and inspiration.


