What's in this repo
Data_Mining_Project.ipynb – the full notebook
Data_Mining_Report.pdf – the detailed write-up
dataset (add separately — see below)
Dataset

Egypt Real Estate Listings, sourced from PropertyFinder.eg via Kaggle (by Hassan Khaled). 19,924 listings — apartments, villas, chalets, and more, across multiple Egyptian governorates. Started with 11 raw features and ended up with about 30 engineered features after preprocessing.

What we did

1. EDA — 5 visualizations: price distribution (heavily right-skewed), property type frequency (apartments dominate), size vs price scatter (moderate positive correlation), payment method breakdown (~80% cash), and top 10 locations (New Cairo, North Coast, and Sheikh Zayed lead).

2. Data Cleaning & Preprocessing — parsed messy strings into usable numbers ("2,500,000 EGP" → 2500000.0, "150 sqft / 120 sqm" → 120.0), extracted maid-room flags from bedroom counts, decomposed delivery dates into month/year, filled missing down payments with 0 (since null meant a cash listing), dropped rows with missing core fields, removed duplicates, one-hot encoded property type and governorate, and used a 3.0 IQR multiplier (instead of the usual 1.5) to remove outliers without cutting out real luxury listings. Numeric features were scaled with StandardScaler, while an unscaled copy was kept for the fuzzy system later.

3. K-Medoids Clustering — chosen over K-Means because it's more interpretable (real data points as centers, not meaningless means for one-hot columns) and more robust to outliers. Used Elbow + Silhouette across K=2–10, picked K=5 (silhouette ≈ 0.1045) to get 5 meaningful market segments instead of an oversimplified 2-way split:

Cluster 0: Affordable Residential
Cluster 1: Mid-Market Apartments
Cluster 2: Premium / Luxury
Cluster 3: Vacation / Chalet
Cluster 4: Suburban Villas

4. Hierarchical Clustering — tried Ward, Complete, and Average linkage on a 2,000-sample subset (full dataset too big for linkage computation). Ward gave the most compact, balanced clusters, and all three confirmed the same K=5 structure found with K-Medoids.

5. Fuzzy Logic System — a Mamdani inference system that scores any property's investment quality from 0–10, using price, size, and bedrooms (unscaled, human-readable values). 8 rules encode intuitive logic (e.g. low price + large size → Excellent; high price + small size → Bad). Validated on 500 random properties with a sensible spread across Bad/Average/Excellent.

6. Genetic Algorithm — feature selection to find the subset of features that maximizes K-Medoids' Silhouette Score, since some governorate dummies add noise rather than signal. Binary chromosome (1 = keep feature), population of 50, 20 generations, tournament selection, single-point crossover, bit-flip mutation, elitism. Improved the silhouette score over using all features, converging by around generation 15–18.

7. Full Pipeline — run_real_estate_pipeline() ties everything together with two parallel pathways that merge at the end:

Expert path: raw values → fuzzy system → investment score
ML path: scale → one-hot encode → GA feature mask → K-Medoids → cluster assignment

Given a property's raw attributes, it outputs a market cluster, an investment score out of 10, and a recommendation label.

Key findings
Price is right-skewed and apartments dominate the market; New Cairo, North Coast, and Sheikh Zayed are the busiest areas.
K=5 balances statistical validity (local silhouette max) with real, interpretable market segments.
The fuzzy system catches bad deals even inside "premium" clusters — e.g. a small, overpriced unit still scores Bad regardless of its cluster.
GA-based feature selection improves clustering by filtering out noisy, rare governorate dummies.
Business value
Investors can screen a property in seconds and get both a market segment and an investment score.
The dual-pathway design means the system stays interpretable (fuzzy rules) while still using data-driven clustering.
The 5-cluster segmentation supports targeted pricing and marketing by tier.
How to run
bash
pip install pandas numpy scikit-learn scikit-learn-extra scikit-fuzzy matplotlib seaborn
