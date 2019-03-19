#  Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for displaying DataFrames

import matplotlib.pyplot as plt

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
plt.show()
