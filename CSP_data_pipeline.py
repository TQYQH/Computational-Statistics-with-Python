import pandas as pd
import numpy as np

# Read the data and check the first 5 rows
# Check if the file is imported successfully
try:
    data_df = pd.read_csv("IPEDS_data.csv")    
except IOError:
    print('Error: No such file or loading failed') 
else:
    print( 'Read successfully')

# Drop the University name column since the ID columns can identify the University
# Check if the column name is existing
try:
    data_df.drop('Name',axis=1,inplace=True)
except Exception:
    print('Error: The column is not exist!')
else:
    print('Delete successfully!')
    
# Delete rows with a value missing rate greater than 40%. 
# Due to too much data loss, it may cause a large error in the analysis results
# First, count the number of the missing value in each row
missing_count=data_df.isnull().sum(axis=1) 
col_count=data_df.shape[1]
# Second,calculate the missing rate of each row of data
missing_ratio=missing_count/col_count 
# Third, add a missing rate column to the dataframe
data_df['missing ratio']=missing_ratio
# At last, drop rows where the missing rate is greater than 40%. 
data_df.drop(data_df[data_df['missing ratio']>0.4].index,inplace=True)

# Transform human language into machine language to facilitate subsequent data processing
# First, check the number of nulls in each column
data_df.isnull().sum()
# Second, convert every 'yes' and 'no' into 1 and 0, respectively
data_df.replace({'No','Yes'},{0,1},inplace=True)

# Take all column names and convert them to list form, prepare for adding column names later
data_df.columns
columns_name=data_df.columns.tolist()

# Use KNN algorithm to fill the missing values
# KNN algorithm fills missing values by analyzing the similarity between rows
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
# n_neighbors 5 just because it is the default value
# It finds the nearest k neighbors based on the value of neighbor and assign an appropriate value to the missing value
imputer = KNNImputer(n_neighbors=5, weights="uniform")
data_df_list = imputer.fit_transform(data_df) 
data_df = pd.DataFrame(data_df_list)

# The column name of the dataframe is dropped since it fills in the missing values in the array type
# Add column names for the dataframe
# The extraction of column names has mentioned before
data_df.columns = columns_name

# Data standardization is to transform each data to have a mean of 0 and a standard deviation of 1.
# For example, in the IPDES data, the values in score columns are much larger than those values in percentage columns
# Thus, use data standardisation to reduce the errors in the subsequent data processing results
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = data_df.iloc[:,2:]
x = scaler.fit_transform(x)
y = data_df.iloc[:,0]

# The IPDES data has too many dimensions, which makes the analysis process difficult.
# Therefore, principal component analysis is used to reduce data dimension
# Reduce the dimension of the data set, while maintaining the features of the data set that contribute the most to the variance
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
pca = PCA().fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# As can be seen, the first 20 components take up 80% variance
# Thus, reduce the data dimensions to 20 and get a new dataframe for further data analysis
pca = PCA(n_components=20)   
pca.fit(x)                  
new_x=pca.fit_transform(x)   
newdata_df=pd.DataFrame(new_x)
newdata_df