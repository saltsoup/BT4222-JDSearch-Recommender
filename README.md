# BT4222-JDSearch-Recommender

BT4222 final project: personalised re-ranking of JD search results.

## 1. Project overview

This project was developed for BT4222 Mining Web Data for Business Insights at the National University of Singapore (AY2026/27 Semester 1).

Given a user's current search query, their interaction history and the list of candidate products that JD's search engine returned, the model re-orders the candidates so that the products the user is most likely to click, add to cart or buy appear first.

The main model is a **Behaviour Sequence Transformer (BST)** that reads the user's action sequence together with 23 engineered features. A **LightGBM LambdaRank** model trained on the same 23 features serves as a more interpretable, white-box comparison and was used to select the features. Both are compared against Random, Popularity, NMF and NCF baselines with one shared scoring code, and two experiments test how much history the ranker needs and which feature families drive its performance.

## 2. Dataset

We use the [JDSearch dataset](https://github.com/rucliujn/JDsearch) of anonymised e-commerce search sessions, product metadata and historical user interactions. The raw files are not committed to this repository; see Section 6 for where the data lives.

| File | Description |
|---|---|
| `user_behavior_data.txt` | Current query, candidate product list, engagement labels and the user's interaction history |
| `product_meta_data.txt` | Product IDs, anonymised product names, brands, categories and shop IDs |

| Label | Meaning |
|---|---|
| 0 | No recorded engagement |
| 1 | Click |
| 2 | Add to cart |
| 3 | Purchase |

## 3. Data preprocessing

| Metric | Result |
|---|---|
| Original sessions | 173,831 |
| Eligible sessions | 171,175 |
| Excluded all-negative sessions | 2,103 |
| Excluded all-positive sessions | 553 |
| Training / validation / test sessions | 136,940 / 17,117 / 17,118 |
| Candidate-level records | 15,348,049 |
| Candidate Parquet chunks | 86 |
| Historical cart actions older than 90 days, excluded from features | 1,177 |

Sessions without both an engaged and a non-engaged candidate were excluded. Candidates were shuffled within each session, because the raw lists store engaged products first, and the 80/10/10 split was drawn by session with a fixed seed (42). Everything learned from labels, such as popularity, is computed from training sessions only.

### Historical logging windows

Exploratory analysis showed that JD recorded each interaction type for a different length of time before the query:

| Time before current query | Interactions recorded |
|---|---|
| 0 to 7 days | Clicks, carts, orders, follows |
| 7 to 90 days | Carts, orders, follows |
| 90 to 365 days | Orders only |

Historical time intervals were reconstructed into elapsed days before the query. The time-window experiment (Section 8) tests what each window contributes.

## 4. Feature engineering

About 90 features were built, grouped into families, and 23 were selected for the final models.

| Family | Description | Examples among the 23 selected |
|---|---|---|
| A. Exact product history | Whether and how recently the user interacted with this exact product | `exact_wid_last_seen_days` |
| B. Brand, shop and category affinity | Time-decayed share of the user's history in the candidate's brand, shop and category | `brand_aff`, `cate3_aff`, `shop_share`, `shop_last_seen_days` |
| C. Title similarity | TF-IDF similarity between the candidate title and the user's history | `title_cos_max`, `title_cos_profile` |
| D. item2vec similarity | Embedding similarity between the candidate and the user's history | `i2v_cos_mean` |
| E. Similarity to the last action | Whether the candidate shares brand or category with the most recent action | `last_same_brand`, `last_same_cate3`, `last_i2v_cos` |
| F. History under the same query | Past interactions with the product under the same query | `qhist_wid_cnt` |
| G. Popularity priors | Smoothed engagement rates among training sessions, out-of-fold | `item_ctr_train`, `cate3_ctr_train` |
| H. Slate-relative features | The candidate relative to the other candidates in the same result list | `cate3_slate_share`, `shop_aff_slate_pct` |
| Q. Current-query match | Token overlap between the query and product name, plus matching indicators for brand and category levels | `name_query_match`, `brand_name_query_match`, `cate_name_1_query_match` to `cate_name_4_query_match`, `query_match_count` |

Every past action is weighted by action type (click 1, follow 2, cart 3, order 5) and an exponential decay with a 30-day half-life. Candidate position is never used as a feature. The full list with descriptions is in `BT4222 Final/JDSearch/processed/feature_dictionary.csv` on the shared Drive (Section 6).

## 5. Repository structure

The repository contains the notebooks needed to generate every intermediate file from the raw data. It does not contain the raw data.

To reproduce the analysis, pull this repo, then transfer `Data Analysis Pipeline/` and `BT4222 Final/` to your Colab-enabled Google Drive at `/content/drive/MyDrive/`, i.e. the root of My Drive. In other words, transfer the **contents** of this repo, **one level below** `BT4222-JDSearch-Recommender/`, into the **root** of your Google Drive.

Then download the raw data and place it in `BT4222 Final/JDSearch/Raw Data/`, named `product_meta_data.txt` and `user_behavior_data.txt`. Their sizes should be 4.51 GB and 912.9 MB respectively, as shown in the Google Drive file details pane. You can download the raw data from the original authors (repository https://github.com/rucliujn/JDsearch, paper https://arxiv.org/pdf/1905.06874) or use the group's shared Drive folder in Section 6, which already contains it together with every intermediate file.

Then open `Data Analysis Pipeline/` and run the notebooks in order.

```
BT4222-JDSearch-Recommender/
├── README.md
├── .gitignore
├── Data Analysis Pipeline/                                 # in pipeline order; each notebook writes its outputs to Drive for the next ones
│   ├── Note_on_file_names.txt                              # map from the original notebook names to the step names
│   ├── step_1_data_preprocessing.ipynb                     # cleaning, 80/10/10 split, candidate chunks, recency features
│   ├── step_2_query_popularity_baseline_models.ipynb       # query-match and popularity features, NMF and NCF baselines
│   ├── step_3_person_match_feature_engineering.ipynb       # user-to-candidate history and similarity features (families A to H), single-feature evaluation
│   ├── step_4_backup_feature_generation.ipynb              # rebuilds the full 90-feature tables from step 3's saved intermediates and writes lgbm_features_all/
│   ├── step_5_build_run_lightgbm_model.ipynb               # feature selection, LightGBM LambdaRank training and prediction
│   ├── step_6_build_train_BST_model.ipynb                  # main Behaviour Sequence Transformer model; Steps 9 and 10 add training predictions and the feature-family experiment
│   ├── step_7_build_window_eval_features.ipynb             # rebuilds the features from 7-day and 90-day history for the time-window experiment
│   ├── step_8_results_and_experiments.ipynb                # shared scoring code, results table, time-window and feature-family experiments
│   ├── step_9_analyse_model_bias.ipynb                     # popularity and exposure bias by bucket, attempted fixes, root cause
│   └── step_10_bst_diagnostics.ipynb                       # user-segment breakdowns, representative failures, exposure-aware re-rank, business insights
└── BT4222 Final/                                           # mirrors the shared Drive folder
    └── JDSearch/
        └── Raw Data/
            ├── README.md                                   # where to download the raw data
            ├── product_meta_data.txt                       # placeholder stub; replace with the real file
            └── user_behavior_data.txt                      # placeholder stub; replace with the real file
```

## 6. Data and large files

The raw dataset, processed features, predictions, model artefacts and result tables are too large for GitHub and are kept in a shared Google Drive folder:

https://drive.google.com/drive/folders/1MdRvvM64-AuWvqBChPhaT21KPxzOkkCj?usp=sharing (anyone with the link can view)

This is the folder the group used to run the pipeline; it contains every artefact and result the report refers to.

The shared folder is `BT4222 Final`, and the notebooks expect it mounted at `/content/drive/MyDrive/BT4222 Final/`. To rerun anything from another Google account without regenerating the data, open the link, right-click the `BT4222 Final` folder, choose "Add shortcut to Drive" with My Drive as the location, then run `drive.mount('/content/drive')` in Colab; the paths in every notebook will resolve as written.

Layout of `BT4222 Final/JDSearch/`:

| Folder or file | Contents | Produced by |
|---|---|---|
| `Raw Data/user_behavior_data.txt`, `product_meta_data.txt` | The JDSearch dataset | JDSearch |
| `Raw Data/product_chunks/` | Product metadata split into Parquet chunks for faster loading | step 2 |
| `processed/user_split.csv`, `data_cleaning_summary.csv`, `feature_dictionary.csv` | Train/validation/test assignment, cleaning counts, feature descriptions | step 1 |
| `processed/candidate_chunks/`, `candidate_chunks_shuffled/` | Candidate-level data in 86 Parquet chunks, before and after within-session shuffling | step 1 |
| `processed/session_recency_features.parquet`, `candidate_recency_chunks/` | Session-level and candidate-level recency features | step 1 |
| `processed/ningjing_candidate_features_checkpoint.parquet`, `ningjing_candidate_features_with_baselines.parquet`, `ningjing_test_features_with_baselines.parquet` | Query-match and popularity features, NMF and NCF scores | step 2 |
| `model_artifacts/nmf_wh.pkl`, `nmf_artifacts.pkl`, `ncf_model.pt` | Saved NMF factors and NCF weights | step 2 |
| `processed/part3_features/` | Feature-pipeline intermediates: session chunks, product tables, item2vec and term2vec vectors, training priors | step 3 |
| `processed/lgbm_features_all/` | Full-year features for train, validation and test, group sizes, feature list and dictionary | step 4 |
| `model_artifacts/lightgbm_popularity_query_search/` | Query-match feature cache, tuned LightGBM models, tuning results, train and test predictions | step 5 |
| `processed/bst_literal/` | BST train and test predictions and best model weights | step 6 |
| `processed/bst_experiments/` | Validation predictions and run summary of the 12 feature-family runs | step 6, Step 10 |
| `processed/lgbm_features_window_7d/`, `lgbm_features_window_90d/` | The same features rebuilt from 7 and 90 days of history | step 7 |
| `model_artifacts/4B_results/` | Results table, experiment tables, feature importance, NMF/NCF score files | step 8 |
| `BST_additional_diagnostics/` | Segment breakdowns, session comparison and the exposure-aware re-rank table | step 10 |

## 7. Reproducibility

The project was developed in Google Colab in Python, using pandas, NumPy, PyArrow, scikit-learn, gensim, LightGBM, PyTorch, matplotlib and seaborn.

To rerun a notebook: open it in Colab, mount Google Drive with the shared folder at the path above, and run the cells in order. Notebooks reuse saved intermediates where they exist, so a rerun of a later notebook does not require rerunning the earlier ones. The feature pipeline (steps 3, 4 and 7) and the BST training (step 6) are the long steps; step 6 needs a GPU runtime, everything else runs on CPU.

Validation rules followed throughout:

- Candidate position is never used as a feature, since the stored order reflects how the candidate lists were constructed.
- Engagement labels are never inputs, and popularity features are computed from training sessions only, out-of-fold for training rows.
- All candidates of a session stay in one split, and history features use only information available before the current query.
- All models are scored with the same functions (Section 2 of step 8, copied into step 10 for the re-rank check), with graded NDCG@10 (click 1, cart 3, purchase 7) and fixed-seed random tie-breaking. Step 10's segment breakdowns additionally report binary NDCG@10 and say so where they do.

## 8. Models and results

Test users (17,118 sessions), graded NDCG@10:

| Model | NDCG@10 | MRR | P@1 |
|---|---|---|---|
| Random | 0.082 | 0.099 | 0.027 |
| Popularity (training users) | 0.255 | 0.268 | 0.158 |
| NMF (+ popularity fallback) | 0.304 | 0.305 | 0.175 |
| NCF (+ popularity fallback) | 0.300 | 0.298 | 0.165 |
| **BST (main model, 23 features)** | **0.385** | **0.394** | **0.258** |
| LightGBM LambdaRank (23 features) | 0.405 | 0.412 | 0.272 |

Both final models beat every baseline on every metric. BST reaches 0.440 on training users against 0.385 on test, while LightGBM (15 leaves, 1,165 iterations) moves from 0.414 to 0.405, so the transformer fits the training histories more closely than it generalises. Full tables with Engagement Recall@10, Purchase Recall@10 and the cold/warm product split are in `main_results_table.csv` under `model_artifacts/4B_results/` on the shared Drive.

Experiments (validation users, models retrained on a quarter of the training sessions; step 8):

- **Time window.** Rebuilding the features from 7 days, 90 days and a full year of history gives NDCG@10 of 0.390, 0.399 and 0.402. Older history helps, and most of the gain comes from days 7 to 90, where carts and orders are still recorded.
- **Feature families.** Removing all personal-history features costs 0.057 (BST) and 0.072 (LightGBM); removing the popularity priors costs 0.030 and 0.052; every other family, including current-query match, costs 0.007 or less. The re-ranker's value comes from the user's history and from popularity rather than from matching the query, which JD's search engine has already done.

Diagnostics (test users; steps 9 and 10):

- **Popularity bias.** Both final models give about 51% of their top-10 slots to products with no training engagement, against 89.5% of candidates, and two inference-time fixes made cold-item recall worse; 2,347 level-3 categories appear only in test candidates (step 9).
- **User segments and exposure.** Both models rank better for users active in the last 7 days, for users with longer histories and for common queries than one-off queries, with LightGBM ahead in every segment. Reserving at least n of BST's top-10 slots for no-engagement products raises their share from 50.9% to 58.4% at n = 3 for a cost of 0.004 NDCG@10 (step 10).

## 9. Limitations

- The three logging windows mean that older history contains only some action types, so action frequencies are not comparable across windows.
- Exact-product history is sparse: 0.78% of candidate rows match a product in the same user's history.
- Some product IDs lack metadata, and 2,347 level-3 category values appear only in test candidates, never in training.
- Sessions without both an engaged and a non-engaged candidate were excluded.
- The experiments use a quarter of the training sessions and a single seed per model, so differences of 0.003 or less in NDCG@10 are not reliable.
- Offline evaluation re-ranks the candidate lists in the log and may not reflect live search behaviour.

## 10. References

- Liu, J. et al. *JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions.* SIGIR 2023. Dataset repository: https://github.com/rucliujn/JDsearch
- Chen, Q. et al. *Behavior Sequence Transformer for E-commerce Recommendation in Alibaba.* 2019.
- He, X. et al. *Neural Collaborative Filtering.* WWW 2017.
- Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Burges, C. *From RankNet to LambdaRank to LambdaMART: An Overview.* 2010.
- Barkan, O. and Koenigstein, N. *Item2Vec: Neural Item Embedding for Collaborative Filtering.* 2016.

## 11. Team

| Member | Responsibility | Notebooks |
|---|---|---|
| Siqian | Data cleaning, train/validation/test split, session chunks, recency features | step 1 |
| Ningjing | Query-match and popularity features, NMF and NCF baselines, popularity and exposure bias diagnosis | steps 2, 9 |
| Aaron | User-to-candidate match features, feature pipeline, BST main model development, repository setup | steps 3, 4, 6 |
| Jingxuan | Feature selection and LightGBM LambdaRank | step 5 |
| Elysia | Shared scoring code, results table, time-window and feature-family experiments | steps 7, 8 |
| Truman | User-segment diagnostics, exposure-aware re-rank and business insights | step 10 |
