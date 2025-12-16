# <ins> Q1: Pick a traditional - non neural network approach to your problem.  Why did you pick this 
approach and how well does a traditional approach do?</ins> 

# i used scikit 
!pip install seaborn
!pip install openpyxl

#importing libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set seaborn style
sns.set(style="whitegrid")



file_path = 'data/VideoGames_Sales.xlsx' #Video Game Sales Data Analysis and Prediction on how to load data
DataFrame = pd.read_excel(file_path)

DataFrame.head()

DataFrame.info()

