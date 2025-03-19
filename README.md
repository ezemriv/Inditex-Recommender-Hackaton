## 📌 Hackathon Overview

This repository contains my solution for a Data Science hackathon focused on **recommender systems** in the e-commerce domain. The primary objective is to develop a model capable of recommending **five personalized products** to users based on their browsing behavior, purchase history, and product attributes.

Participants were provided with interaction data (sessions, products, users), and the final system is evaluated on its ability to **rank products effectively**, using **Normalized Discounted Cumulative Gain (NDCG)** as the evaluation metric. The challenge aims to simulate real-world retail scenarios where user preferences and external trends influence product selection, especially in the fashion sector.

Here’s the updated section incorporating the reference to the `notebooks/` folder, the use of **Polars** for data processing, and **Kaggle** for model training and inference:

---

## 📄 Hackathon Instructions

All official hackathon instructions, dataset descriptions, and evaluation criteria are detailed in [**Hackaton_instructions.md**](./Hackaton_instructions.md).

---

## 🧩 My Solution

To address the recommendation task, I performed **extensive feature engineering** combining user-level, product-level, session-level, and interaction-based features. I trained a **LightGBM Ranker** model on the enriched training data to predict product relevance within each session.

Due to the size of the interaction data, most preprocessing and feature engineering were done efficiently using **Polars**. Model training and prediction were executed on **Kaggle** to leverage more powerful compute resources.

For the test set, I generated diverse **candidate product combinations** per session, including:
- Globally popular products.
- Popular products by user country.
- Products from the user’s previous interactions.
- Products most similar (via embeddings) to those the user added to cart.

After scoring these candidates, the top 5 products with the highest predicted relevance were selected as recommendations per session.

Detailed implementation and analysis can be found in the [**notebooks/**](./notebooks/) directory.