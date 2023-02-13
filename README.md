# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Civic data analysis

## Problem Statement

There are many factors which influence the price of a home. Some factors are well known, such as the interior area, yard size, and neighborhood, but these only represent a small fraction of the attributes that are taken account when pricing a home. As such, creating an accurate list price for a house that's being put on the market is a very difficult task, and innacurate appraisals can be costly. If the appraisal is too high, homeowners may struggle too sell, and if it is too low they, as well as their real estate agent, will lose money in the transaction. 

Therefore, a tool which can determine if a home has a reasonable price would be quite valuable. I will use linear regression techniques to assess the prices of homes in Ames, Iowa sold between 2006-2010 and assess whether those homes were priced fairly. Potential home buyers could use the price prediction to determine if a house is worth the money, and realtors could estimate the price of a home to aid them in the appraisal.

## Data Dictionaries

[Data Dictionary here](https://www.kaggle.com/competitions/1031-ames-competition/data)

## Analysis

This analysis consisted of a large grid search of over 50 thousand combinations of hyperparameters. For each combination, R^2 and RMSE were calculated and stored. The best model was chosen as the configuration which yielded the highest cross-validation score: a Lasso regressor with lambda=0.00497.

Unfortunately, time constraints limited the analysis I was able to perform on the different model configurations. Therefore, the model selected was a strong contender for being the ideal model for the problem statement, but there were many, many more which could have performed just as well if not better.

I was not able to come to a full understanding of the effects of each hyperparameter in this study, but I intend to in future analysis. Most notably of interest would be to study the exact effects predicting price per area instead of total price, and more strict feature screening. 

<p align="center">
  <img width="600" height="500" src="https://git.generalassemb.ly/alexwalshml/project_2/blob/main/img/residuals.png">
</p>

For the training set, the model fit excellently, with an R^2 = 0.937. A strong linear relationship exists between the predicted and true values, indicating that the fit accurately represents the examples in the training set. However, when tested on new data, the RMSE inscreased from $14,401 to $36,671. Both of these RMSE values are still a considerable improvement over the baseline value of $57,178.

## Conclusions and Recommendations

In conclusion, the grid search techniques used in model selection, as well as the automated feature selection, proved very powerful for creating a fit for the training data. Comparing the fit to validation and test data, the model does generalize to newer data, but not especially well. Therefore, this model should only be used as a supplement and not in lieu of professional appraisal. It will save time and effort, but is prone to mistakes in its current state. 

Due to time constraints, the full breadth of the modeling techniques was not explored. The information included in the model statistics file is quite likely enough to know where to start on creating a model that both performs well and doesn't overfit. Thus, after implementing a grid search, the results should be meticulously studied, much more so than was done in this analysis.

Additional study should also attempt to implement polynomial features, as they would be able to effectively capture multiplicative relationships between features.

