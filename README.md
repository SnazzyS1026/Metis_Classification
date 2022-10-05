# Classification - Predicting NBA Rookie Longevity
The goal of this project was to iterate through various ML models to find the best performing one.  The final model would be able to predict whether an NBA rookie would play beyond 4 years (a typical rookie-yr contract length is 1-4 years) based on their first-year stats.  The process included running various models, such as: kNN, Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost. Exploration into rebalancing class distributions, log transformation of features, tuning hyperparameters, and feature reduction were done in order to improve the scoring metric.
## Data
Data was webscraped off of the NBA official website and includes all seasons from 1996-97 to 2021-22 (current).  There were over 12K+ data points with over 20+ features in the raw dataset.  A more compressed dataset was created with only the first-year stats of each NBA player along with these added columns: NUM_YEARS (for the number of years that the player has played in total in the NBA thus far), and TARGET_4UP ("0" for playing 4 or fewer years, "1" for playing more than 4 years).  The latter dataset contained just over 2,400 data points. Because of the smaller dataset, stratified k-fold cross validation and stratification were utilized  to make the model more robust.
## Class Imbalance
To address the slight imbalance in class distributions, random oversampling was done to increase the minority class (target class) to level with the majority class.
## Feature Engineering
Through EDA, many of the features were found to be left skewed.  For us not to violate any of the normality assumptions of Gaussian NB, log transformation was performed on the top 5 features.  Scores for the Gaussian NB did indeed rose nicely after the transformation. However, Gaussian NB was ultimately not the final model chosen, and log transformation was then unnecessarily making our model more complex. Consequently, no log transformation was done on the features in the final model.
## Tuning Hyperparameters
RandomizedSearchCV was first to narrow down ranges of values for optimizing hyperparameters.  GridSearchCV was then used to pinpoint the set of the most optimal hyperparameters to maximize scores, model efficiency, and minimize overfitting.
## Feature selection
With over 20+ features, feature selection was used to reduce the complexity of our model, and likely some overfitting as well.  After feature selection, 11 were deemed important for prediction.
# Final model
The best performing model was found to be Random Forest.  For a more detailed look into the entire process, refer to the notebooks attached.
