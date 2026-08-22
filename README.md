# Socio-Economic Drivers of High Income 
### An analysis of the UCI Adult dataset using Unsupervised & Supervised Learning
![alt text](<Income image-1.jpg>)

## 1. Introduction
An interactive Streamlit application that explores socioeconomic factors associated with income, predicts whether an individual earns above $50K, and segments individuals using K-Means clustering.
Income inequality remains one of the most persistent socio-economic challenges globally. While it is often discussed in terms of wages alone, income is actually shaped by a complex interplay of geography, education, occupation, and life-stage factors.

This project uses the UCI Adult Census dataset to unpack these drivers at two levels. At the macro level, we ask: do countries provide different environments for income growth? At the micro level, we ask: within those environments, who are the individuals most likely to earn above $50K and why?

By combining unsupervised clustering to discover hidden economic groups and supervised classification to predict high earners, this analysis moves beyond simple correlation to map the full pipeline from national resources to individual income outcomes.

## 2. Live Demo
View the live application here (https://census-income-predictor.streamlit.app/)

## 3. Data Description
- **Source:** UCI Adult Census Dataset (32,561 records)
- **Target:** Income: `>50K` vs `<=50K`
- **Key Features:** age, education-num, occupation, marital-status, hours-per-week, capital-gain/loss, native-country

## 4. Objectives
- Explore demographic and socioeconomic patterns in the dataset
- Identify factors associated with earning more than $50k 
- Build a machine learning classification model
- segment individuals using K-Means clustering
- Explore income patterns across countries
- Present findings through an interactive Streamlit dashboard
- 
## 5. Data Wrangling

The raw UCI Adult dataset required extensive cleaning before modeling.
During the data wrangling process, thorough checks were conducted on the UCI Adult dataset to identify and address data quality issues. This included:

**1. Handling Missing Values:** 
- The dataset had missing values represented by `?`. These were found in 3 columns: workclass, occupation, and native-country. I removed all rows that contained `?` and also removed duplicate rows.


**2. Feature Engineering:**
- Created `education-years` from education-num for numeric analysis
- Grouped `native-country` into aggregated country-level features (avg education, % >50K, avg capital) for macro clustering
- Binned `capital-gain` and `capital-loss` to handle extreme skewness
- Encoded `income` as binary: 0 for <=50K, 1 for >50K

**3. Encoding & Scaling:**
- One-Hot Encoding for nominal features (workclass, marital-status, occupation, race)
- Label Encoding for ordinal education
- StandardScaler applied to numeric features (age, hours-per-week, capital) for KMeans clustering

**4. Handling Class Imbalance:**
- The target is imbalanced (75.9% <=50K vs 24.1% >50K)
- Applied **SMOTEENN** (SMOTE + Edited Nearest Neighbours) to oversample minority class and clean noisy samples, which improved recall for high earners from 45% to 75%.

These steps enhanced the reliability of the dataset and ensured its suitability for clustering and classification.

## 6. Dependencies
For the project to run, the following needs to be installed and imported
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- scikit-learn
- imbalanced-learn
- Streamlit
- GitHub

## 7. How to run the app 

git clone https://github.com/OluwabukolaAwolusi/census.git
cd "C:\Users\DELL\OneDrive\Dokumente\census\census"
pip install -r requirements.txt
streamlit run app.py

## 8. Project Files
The project in this repository resides in "income prediction(adult dataset).ipynb" file.

## 9. Exploratory Data Analysis (EDA)
Key Insights:
Education -> Occupation: HS-grad and below concentrate in Service / Manual labor. Bachelors+ concentrate in Exec-managerial (47.9% >50K) and Prof-specialty (45.0% >50K).
Marital status: Married individuals have 44.5% >50K rate vs 4.81% never-married and 9.74% previously-married.
Age & Capital Matter: Income rises with age, hours worked, and capital gains.


**Methodology**
Clustering - Macro Level (Countries):
Aggregated data by country
KMeans, k=2
Result: Higher_Resource (North America, Western Europe) vs Low_Resource
Higher_Resource has 3.3x higher >50K rate (32.77% vs 9.92%), 28.7% more education, 2.17x capital gain

The Higher-Resource cluster (Red) - concentrated in North America & Western Europe - has 3.3x higher rate of >50K income (32.77% vs 9.92%).

*Full country stats table is in the notebook.*
![alt text](<Screenshot 2026-08-11 132611.png>)


Clustering - Micro Level (Individuals):
Features: age, hours-per-week, capital-gain/loss
KMeans, k=3
Cluster 0 - Nine-to-Five (87%): 37.9 yrs, 40.4 hrs, 0 capital gain. Only 19.7% >50K
Cluster 1 - Capital-gain-winners (8.4%): 43.7 yrs, 43.9 hrs, 8.84 avg capital gain. 62.7% >50K
Cluster 2 - Risk-Takers (4.7%): 41.8 yrs, 43.6 hrs, 7.51 avg capital loss, 0 gain. 51.3% >50K

![Clusters Profile](images/clusters_profile.png)

Classification:
Model: RandomForest + SMOTEENN for imbalance
Best Threshold: 0.677
Performance: 84% Accuracy, Precision 0.65 / Recall 0.75 / F1 0.70 for >50K class
Highly effective at identifying potential high earners (75% recall).

![Confusion Matrix](images/confusuion%20matrix(RF%40%200.50%20AND%20RF%40%200.677).png)

## 10. Conclusion
Income inequality exists at 2 levels and they reinforce each other.
Macro Level: Countries cluster into "Higher Resource" and "Lower Resource" based on education and capital. This sets the environment.
Micro Level: Within any country, individuals cluster into 3 personas based on life stage and financial behavior. "Established" individuals with age, hours, and assets are 3x more likely to earn >50K than "Early-Career" individuals.
Country Resources → Education → Occupation → Age/Capital → Cluster → Income >50K. 
The classification model confirms we can predict high earners with 75% recall, and the biggest features driving it are education, occupation, age, and hours — all factors that define the clusters above.


## 11. Recommendations
For Business / Marketing:

Cluster 0 (Upskilling ROI): Offer training to Prof-specialty + financial literacy.
Cluster 1 (Premium): Target with wealth management & investment products.
Cluster 2 (Risk Mitigation): Offer insurance, income smoothing, business banking.
For Policy:

Education first: HS-grad to Bachelors has highest impact.
Formalize entrepreneurship: Self-emp-inc earns 2x Self-emp-not-inc.
Country Level: Low-Resource countries need asset-building programs. High-Resource countries need retention & upskilling to Exec-managerial roles.

## 12. Limitations
1. Data Age & Generalizability
The Adult Census dataset is from the 1994 US Census. Income patterns, occupation distribution, and work culture have significantly changed in 30 years. Findings may not generalize to current global income dynamics or modern gig economy jobs.

2. Class Imbalance Handling
Only ∼24% of individuals earn >50K, leading to an imbalanced dataset. To address this, SMOTEENN (a hybrid of SMOTE oversampling and Edited Nearest Neighbours cleaning) was applied to the training set after train_test_split to avoid data leakage. While this improved minority class recall, it also introduced synthetic samples that may not perfectly represent true income distribution, and the Precision-Recall trade-off remains (PR Curve AP = 0.754). The baseline RandomForest without resampling showed lower recall for the >50K class.

## 13. Future Work
1. Model Interpretability: Implement SHAP values to explain individual predictions, especially for the Capital_Gain_Winners cluster, to move beyond black-box Random Forest.

2. Advanced Modeling: Compare current Random Forest with Gradient Boosting (XGBoost, LightGBM, CatBoost) and calibrated classifiers to improve PR-AUC beyond 0.754.

3. Updated Dataset: Re-train on more recent income datasets which reflect post-pandemic income distribution and remote work patterns.

   
## 14. License
This project is licensed under the MIT License.
