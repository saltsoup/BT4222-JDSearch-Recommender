# BT4222-JDSearch-Recommender
BT4222 final project: JDSearch personalised product re-ranking

## 1. Project Overview

This project is developed for **BT4222: Mining Web Data for Business Insights** at the National University of Singapore (AY2026/27 Semester 1).

We aim to develop a personalized product re-ranking system using the JDSearch dataset. Given a user's current search query, historical interactions, and a list of candidate products, the model predicts product relevance and reorders candidates to prioritize products with higher expected engagement.

Our approach combines:
- Search query and product metadata features
- Historical user interaction and recency features
- Product popularity and personalized preference signals
- Machine learning-based ranking models

The project investigates challenges including sparse user-item interactions, cold-item coverage, historical logging windows, and data leakage.

## 2. Dataset

We use the **JDSearch dataset**, containing anonymized e-commerce search sessions, product metadata, and historical user interactions.

### Main data sources

| File | Description |
|---|---|
| `user_behavior_data.txt` | Current queries, candidate product lists, engagement labels, and historical user interactions |
| `product_meta_data.txt` | Product IDs, anonymized product names, brands, categories, and shop IDs |

### Engagement labels

| Label | Meaning |
|---|---|
| 0 | No recorded engagement |
| 1 | Click |
| 2 | Add to cart |
| 3 | Purchase/order |

## 3. Data Preprocessing

The preprocessing pipeline prepares the raw JDSearch data for model training and evaluation.

### Cleaning summary

| Metric | Result |
|---|---:|
| Original sessions | 173,831 |
| Eligible sessions | 171,175 |
| Excluded all-negative sessions | 2,103 |
| Excluded all-positive sessions | 553 |
| Training sessions | 136,940 |
| Validation sessions | 17,117 |
| Test sessions | 17,118 |
| Candidate-level records | 15,348,049 |
| Candidate Parquet chunks | 86 |
| Historical cart actions excluded from engineered features | 1,177 |

Sessions without both positive and negative candidate labels were excluded from the current ranking experiment.

Candidates were shuffled within each session while preserving their product IDs and labels. A fixed random seed (`42`) was used for the 80/10/10 train-validation-test split.

### Historical interaction windows

Exploratory analysis revealed different apparent recording windows for historical interaction types:

| Time before current query | Main recorded interactions |
|---|---|
| 0–7 days | Clicks, carts, orders, follows |
| 7–90 days | Carts, orders, follows |
| 90–365 days | Primarily orders |

Historical consecutive time intervals were reconstructed into elapsed days before the current query.

The 1,177 cart interactions recorded more than 90 days before the query were excluded from engineered history features, while the original raw data was preserved.

## 4. Feature Engineering

The project uses several feature groups.

| Feature group | Description | Status |
|---|---|---|
| Session recency | Counts of historical actions across time windows; history length and recency | Completed |
| Exact-product recency | Candidate-specific historical clicks, carts, orders and last interaction time | Completed |
| Query-product matching | Overlap between search terms and product metadata | In development |
| Popularity | Historical popularity of products, brands and categories | In development |
| Personal preferences | Historical user affinity for brands, categories and shops | In development |

### Key preprocessing findings

- **99.53%** of sampled unique candidate IDs matched product metadata.
- **92.94%** of sampled unique historical product IDs matched product metadata.
- **120,235 of 15,348,049 candidate rows (0.78%)** matched a product in the same user's recorded history.

The low exact-product match rate motivates the use of broader product attributes and historical preference features.

## 5. Repository Structure

```text
BT4222-JDSearch-Recommender/
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── [Additional modelling notebooks]
├── docs/
│   ├── feature_dictionary.csv
│   └── data_cleaning_summary.csv
├── src/
│   └── [Reusable scripts, if applicable]
└── results/
    └── [Evaluation tables and figures]
```

Large raw datasets and processed Parquet files are stored separately in Google Drive and are not committed to GitHub.

## 6. Processed Data Outputs

The preprocessing notebook generates the following files under:

`BT4222 Final/JDSearch/processed/`

| File / Folder | Purpose |
|---|---|
| `user_split.csv` | Reproducible train/validation/test assignments |
| `candidate_chunks_shuffled/` | Candidate-level data, split into 86 Parquet chunks |
| `session_recency_features.parquet` | Session-level historical interaction features |
| `candidate_recency_chunks/` | Candidate-specific historical interaction features, aligned with shuffled chunks |
| `feature_dictionary.csv` | Feature descriptions and types |
| `data_cleaning_summary.csv` | Preprocessing counts and cleaning decisions |

## 7. Reproducibility

### Environment

The project is developed primarily in **Google Colab**, using Python.

Main libraries include:
- pandas
- NumPy
- PyArrow
- scikit-learn
- tqdm

Additional modelling dependencies will be documented as the project progresses.

### Running the preprocessing pipeline

1. Open `notebooks/01_data_preprocessing.ipynb` in Google Colab.
2. Mount Google Drive.
3. Ensure the JDSearch raw datasets are available in the expected project directory.
4. Run the notebook cells sequentially.
5. Verify the saved split, cleaning summary, and Parquet outputs.
6. Use the processed outputs for feature engineering and model training.

The notebook expects the following directory:

`/content/drive/MyDrive/BT4222 Final/JDSearch/`

Users running the notebook from another Google account may need to update the data paths.

### Important validation precautions

- Candidate position must not be used as a predictive feature.
- Engagement labels must not be included as input features.
- Popularity features must not incorporate held-out evaluation outcomes.
- All candidates belonging to the same session must remain in the same split.
- Candidate-specific feature chunks must be aligned with their corresponding candidate chunks.
- Historical features must use only information available before the current query.

## 8. Models and Evaluation

The project compares simple recommendation baselines with a feature-based ranking model.

Planned or existing model families include:
- Popularity-based ranking
- Neural Collaborative Filtering (NCF)
- Non-negative Matrix Factorization (NMF)
- LightGBM learning-to-rank

Evaluation metrics include:
- NDCG@10
- Mean Reciprocal Rank (MRR)
- Precision@1
- Engagement Recall@K
- Purchase Recall@K

Final model configurations and verified results will be added after training and evaluation.

## 9. Limitations

- Historical interaction types have different apparent recording windows, limiting direct comparison of action frequencies across time.
- Exact-product interaction history is sparse.
- Some historical product IDs lack matching product metadata.
- The current ranking experiment excludes sessions without both positive and negative candidates.
- Offline evaluation uses the provided candidate sets and may not directly represent production search-result ranking.

## 10. References

- JDSearch dataset paper and official repository.
- Additional modelling references will be added as the project progresses.

## 11. Team Contributions

| Role | Responsibility |
|---|---|
| Person 1 | Data cleaning, train/validation/test split, recency features, repository setup |
| Person 2 | Query matching and popularity features |
| Person 3 | Personalized preference features |
| Person 4A | LightGBM model training |
| Person 4B | Evaluation metrics and experiments |
| Person 5 | Diagnostics and business insights |

Team member names and final contributions will be updated before submission.
