# Heart Disease Diagnosis

This project aims to develop an MLops Pipeline for Bias Detection and Mitigation in Heart Disease Diagnosis Models[^1^][1].

## Problem Statement

Heart disease is one of the leading causes of death worldwide, affecting millions of people every year. However, not all people have the same risk of developing heart disease, as there are various factors that influence the likelihood of having a heart attack, such as age, gender, blood pressure, cholesterol, diabetes, smoking, etc. 

Therefore, it is important to have accurate and reliable models that can diagnose heart disease based on these factors, and provide timely and appropriate treatment to the patients. However, these models may also suffer from bias, meaning that they may not perform equally well for different demographic groups, such as different age groups or genders. This may lead to unfair and inaccurate predictions, and potentially harm the patients who belong to the disadvantaged groups.

Hence, the goal of this project is to create a pipeline that can detect and mitigate bias in heart disease diagnosis models, and ensure that they provide equitable predictions for diverse demographics[^2^][2].

## Approach

The approach of this project consists of the following steps:

- Start: We use the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) as the input dataset, which contains 14 attributes and 303 instances related to heart disease diagnosis.
- Identified Demographic Groups: We identify two demographic groups in the dataset, namely age and gender, and analyze their distribution and impact on the target variable (presence of heart disease).
- Reweighing algorithm to mitigate bias and obtain training data: We use the [AI Fairness 360 toolkit](https://aif360.mybluemix.net/) to apply the reweighing algorithm, which is a preprocessing technique that assigns weights to the instances in the dataset, such that the weighted dataset has reduced disparity between the demographic groups. We use the weighted dataset as the training data for the models.
- Ensemble Model (Voting Classifier): We use three different classification models, namely Decision Tree, Random Forest, and Gradient Boosting, and combine them into an ensemble model using the voting classifier technique, which takes the majority vote of the individual models as the final prediction. We use the accuracy metric to evaluate the performance of the models.
- Prediction of Results: We use the ensemble model to predict the presence of heart disease for new instances, and compare the results with the actual outcomes.
- Deployment of Pipeline to AWS and Github Actions: We deploy the pipeline to AWS and Github Actions, which allows us to automate the model training, testing, and deployment process, and monitor the model performance and bias metrics over time.

## Models

We use the following models in our ensemble model:

- Decision Tree Model: We control the model complexity with a maximum depth of 5 to prevent overfitting[^3^][3]. We also enhance the robustness by setting a minimum of 10 samples required for a node split[^4^][4].
- Random Forest Model: We employ an ensemble of 100 decision trees to prevent overfitting and enhance accuracy[^5^][5]. We also limit the individual tree depth to 5 for a well-balanced model[^6^][6].
- Gradient Boosting Model: We utilize 100 weak learners with a learning rate of 0.1 to prevent overfitting during model training[^7^][7].

The accuracy of the ensemble model is 95%.

## Conclusion

- The MLops pipeline with ensemble techniques ensures unbiased Heart Disease diagnosis models, promoting equitable predictions for diverse demographics[^2^][2].
- The deployment demonstrates practical implementation of ethical healthcare solutions, emphasizing transparency and accountability[^8^][8].
- The collaboration and innovation among participants underscore the collective drive toward fair and reliable healthcare models[^9^][9].
