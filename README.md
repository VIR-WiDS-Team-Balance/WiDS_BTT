### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| Fatima Asif | @fatimasif | Implemented data visualizations and basic feature engineering |
| Maria Antonov | @mariaantonov | Implemented the initial training and testing pipeline for model development |
| Cheyenne Bajani | @cheyennebejj123 | .. | Implemented models to evaluate features 
| Daphney Talekar | @daphneyt04 | set up  |

## **🎯 Project Highlights**

*Built a CNN-based deep learning model using 3D MRI scan data to predict ADHD diagnosis
* Achieved an F1 score of .42 and a ranking of 513 on the final Kaggle Leaderboard
* Used \[explainability tool\] to interpret model decisions
* Implemented \[data preprocessing method\] to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**


Clone the Environment
Set Up Repository
We recommend using Google Colab for running this project. All dependencies are handled within the notebook.
Join WiDS competition / download datasets from WiDS competition page or datasets provided within the repository

---

## **🏗️ Project Overview**

* The WiDS competition challenged participants to use structural brain imaging data to predict whether an individual has ADHD. Through the Break Through Tech AI program which moves to support underrepresented individuals within the technical space, the WiDS serves as a complementary mission to use technology for good and serve others.
* The objective of the challenge is to develop machine learning models that can accurately classify whether an individual has Attention Deficit Hyperactivity Disorder (ADHD) using structural brain imaging data.
* Accurate classification of ADHD through imaging could potentially reduce delays in diagnosis and provide early intervention. Our solution contributes to the broader goal of identifying neurological patterns and promoting fairness in AI-assisted diagnostics.

---

## **📊 Data Exploration**

Plots, charts, heatmaps, feature visualizations, sample dataset images
![image](https://github.com/user-attachments/assets/6343eea1-d83c-4cdc-8194-1ddd492e6fbf)
![image](https://github.com/user-attachments/assets/4b7971ed-846e-4c14-8142-7c95ac73184a)

---

## **🧠 Model Development**
Our approach involved developing two separate machine learning models to predict ADHD diagnosis and sex classification using functional brain imaging data, sociodemographic information, emotions, and parenting data. 

 Model(s) used:
We implemented Random Forest Classifiers for both tasks due to their robustness is handling complex datasets, including numerical and categorical features. Random Forest models are well-suited for multi-feature datasets, as they handle non-linearity, feature interactions, and missing values efficiently.

 Feature Selection and Hyperparameter Tuning
- Feature Engineering:
- Included quantitative metadata (e.g., EHQ total score, color vision score).
- Selected functional connectome features** (brain connectivity patterns).

Preprocessing
Numerical features were imputed using the median and standardized using StandardScaler() from sklearn.preprocessing.

Hyperparameter Tuning
The model used class_weight='balanced' to address class imbalance, particularly for ADHD and female cases.

A default of 100 estimators (n_estimators=100) was used in the Random Forest Classifier without extensive tuning.

Future improvements could include hyperparameter optimization using tools like GridSearchCV or RandomizedSearchCV.

Training Setup
Data Split: The dataset was split into 80% training and 20% validation using train_test_split() from sklearn.model_selection.

Model Training: Two separate Random Forest models were trained on the same dataset to investigate potential sex-based differences.

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---Validation Accuracy: ~0.82

F1 Score (macro): ~0.79

AUC-ROC: 0.85

These results suggest the model is reasonably effective at identifying ADHD cases from the feature set.



## **🖼️ Impact Narrative**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?

1. What brain activity patterns are associated with ADHD? Are they different between males and females?
Preliminary findings suggest that certain functional brain connectivity features are predictive of ADHD. These patterns may differ by sex, supporting the hypothesis that females may present with distinct neurobiological markers of ADHD compared to males. This could explain why ADHD in females is often underdiagnosed or mischaracterized.

2. How could your work help contribute to ADHD research and/or clinical care?
Early Identification of At-Risk Individuals: Machine learning models using neuroimaging and demographic data could flag individuals—especially females—who may exhibit ADHD traits not captured by current clinical standards.

Personalized Medicine: Sex-specific brain connectivity patterns could lead to tailored treatment plans, enhancing the effectiveness of interventions.

Reducing Diagnostic Bias: By highlighting structural and functional differences in ADHD presentation across sexes, this work supports more inclusive diagnostic criteria and encourages greater clinical awareness.

---

## **🚀 Next Steps & Future Improvements**

Model Limitations
🔹 Limited Feature Set: The model does not yet include categorical socio-demographic features that could be informative.

🔹 Class Imbalance: Despite using class_weight='balanced', fewer ADHD cases in females may lead to reduced sensitivity for this group.

🔹 Model Complexity: Random Forests are relatively simple; they may not fully capture nonlinear or hierarchical patterns in neuroimaging data.

🔹 Overfitting Risk: Due to the modest dataset size, there's a possibility of overfitting despite tree-based regularization.

With More Time/Resources, We Would:
 Feature Engineering: Explore interactions between demographic and connectivity features.

Hyperparameter Optimization: Use GridSearchCV or RandomizedSearchCV to fine-tune model parameters.

Cross-Validation: Employ k-fold cross-validation for more robust evaluation and better generalization.

 Advanced Models: Explore Gradient Boosting, Neural Networks, or AutoML platforms for improved performance.

Additional Datasets or Techniques to Explore
 Multi-modal Neuroimaging: Incorporate fMRI, DTI, or structural MRI for richer brain representations.

Deep Learning: Use CNNs for imaging data or RNNs for time-series patterns in connectivity.

 External Datasets: Apply the model to larger, more diverse cohorts to evaluate generalizability and fairness across populations.

---

## **📄 References & Additional Resources**

ADHD Neuroimaging & Machine Learning
Sato, J.R., et al. (2012). “Evaluation of pattern recognition and feature extraction methods in ADHD prediction.” Frontiers in Systems Neuroscience.
https://doi.org/10.3389/fnsys.2012.00068

Functional Connectivity & ADHD
Cao, M. et al. (2014). “Topological organization of the human brain functional connectome across the lifespan.” Developmental Cognitive Neuroscience.
https://doi.org/10.1016/j.dcn.2014.02.004

Sex Differences in ADHD Brain Structure
Fair, D.A. et al. (2012). “Neurophysiological evidence for a male-biased connectivity architecture in ADHD.” Biological Psychiatry.
https://doi.org/10.1016/j.biopsych.2011.09.024
