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
plt.show()

#Create a new dataframe containing only pH and fixed acidity columns to visualize their co-relations
fixedAcidity_pH = data[['pH', 'fixed acidity']]

#Initialize a joint-grid with the dataframe, using seaborn library
gridA = sns.JointGrid(x="fixed acidity", y="pH", data=fixedAcidity_pH, height=6)

#Draws a regression plot in the grid
gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})

#Draws a distribution plot in the same grid
gridA = gridA.plot_marginals(sns.distplot)
