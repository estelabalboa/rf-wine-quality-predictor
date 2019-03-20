#  Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the Red Wines dataset
data = pd.read_csv("data/winequality-white.csv", sep=';')

# Display the first five records
display(data.head(n=2))

data.isnull().any()

data.describe()

data.info()

# Some more additional data analysis
display(np.round(data.describe()))

# La linearidad nos da la relacion entre las variables , feature importance
# alpha permite discretizar los datos.
pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(11, 11), diagonal='kde')

# dimensions_feature , ver si podemos representar relación entre n>2 variables

correlation = data.corr()
#display(correlation)
plt.figure(figsize=(14, 12))
import seaborn as sns
# matplotlib como en esteroides
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

#Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, height=6)

#Draws a regression plot in the grid
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.distplot)


fixedAcidity_citricAcid = data[['citric acid', 'fixed acidity']]
grid_a= sns.JointGrid(x="fixed acidity", y="citric acid", data=fixedAcidity_citricAcid, height=6)
grid_a = grid_a.plot_joint(sns.regplot, scatter_kws={"s": 10})
grid_a = grid_a.plot_marginals(sns.distplot)


volatileAcidity_quality = data[['quality', 'volatile acidity']]
g = sns.JointGrid(x="volatile acidity", y="quality", data=volatileAcidity_quality, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)


#We can visualize relationships of discreet values better with a bar plot

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='volatile acidity', data=volatileAcidity_quality, ax=axs)
plt.title('quality VS volatile acidity')

plt.tight_layout()
plt.show()
plt.gcf().clear()


quality_alcohol = data[['alcohol', 'quality']]
g = sns.JointGrid(x="alcohol", y="quality", data=quality_alcohol, height=6)
g = g.plot_joint(sns.regplot, scatter_kws={"s": 10})
g = g.plot_marginals(sns.distplot)

fig, axs = plt.subplots(ncols=1,figsize=(10,6))
sns.barplot(x='quality', y='alcohol', data=quality_alcohol, ax=axs)
plt.title('quality VS alcohol')

plt.tight_layout()
plt.show()
plt.gcf().clear()
# plt.show()

# For each feature find the data points with extreme high or low values
for feature in data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(data[feature], q=25)

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(data[feature], q=75)

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range

    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])

# OPTIONAL: Select the indices for data points you wish to remove
outliers = []

# Remove the outliers, if any were specified
good_data = data.drop(data.index[outliers]).reset_index(drop=True)

#Displays the first 2 columns
display(data.head(n=5))

# Split the data into features and target label
quality_raw = data['quality']
features_raw = data.drop(['quality'], axis=1)

# Import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
                                                    quality_raw,
                                                    test_size=0.1,
                                                    random_state=42)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Import any three supervised learning classification models from sklearn
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LogisticRegression

# Initialize the three models
reg = RandomForestRegressor(min_samples_leaf=9, n_estimators=100)
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

from sklearn.metrics import r2_score
sco = reg.score(X_test, y_test)
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)
print("Score : ", sco)


# Import a supervised learning model that has 'feature_importances_'
model = RandomForestRegressor(min_samples_leaf=5, n_estimators=250)

# Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_
importances = model.feature_importances_

print(X_train.columns)
print("Importances", importances)

# TODO: plot as an histogram using fastai library
