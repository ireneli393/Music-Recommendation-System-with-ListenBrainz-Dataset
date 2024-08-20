# Recommendation-System-with-ListenBrainz-Dataset

**Project Overview:**

This project focuses on building recommendation systems using the ListenBrainz Dataset, which contains 21,684 users and 695 million playtime records. The dataset is divided into three tables: tracks, interactions, and users, with our analysis centered on the tracks and interactions datasets.

**Data Preparation:**

- We worked with a smaller version of the dataset from 2018, joined the tracks and interactions datasets on `user_id`, and cleaned the data by replacing missing values and filtering out tracks with fewer than 10 plays.
- To avoid the cold start problem, we partitioned the dataset by user and split it into training (70%) and validation (30%) sets using window functions.

**Modeling and Evaluation:**

1. **Alternating Least Squares (ALS) Model:**
   - Implemented using PySpark, the ALS model learns latent factors for users and tracks based on listen counts.
   - We tuned three parameters—`rank`, `regParam`, and `alpha`—and found that `rank=200`, `regParam=0.1`, and `alpha=1` provided the best results with a validation MAP of 0.061372 and precision at 100 of 0.129533.
   - The model was applied to the test set, achieving a MAP of 0.05311.

2. **LightFM Model:**
   - We used the LightFM Python library for collaborative filtering, focusing on improving performance by incorporating metadata.
   - The LightFM model, using the same dataset partitioning, achieved higher precision at k=100 compared to the ALS model with faster training times, making it more ideal for smaller datasets.

**Key Metrics:**
- **Mean Average Precision (MAP):** Evaluates the order of recommended items.
- **Precision:** Measures the accuracy of recommendations.
- **RMSE:** Indicates the fit of the ALS model to the training data.

**Findings:**
- The ALS model performed well on smaller datasets, but LightFM provided slightly better precision with faster training.
- LightFM may be more suitable for smaller datasets, while ALS might be better for larger datasets, though this was not tested due to resource limitations.

**Conclusion:**
This project demonstrates the application of collaborative filtering using ALS and LightFM models on the ListenBrainz dataset, highlighting the trade-offs between model complexity and computational efficiency.

