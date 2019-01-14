# Ebru-Cangul
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

train=pd.read_csv('C:/Users/asus/Desktop/python project/train.csv')
test=pd.read_csv('C:/Users/asus/Desktop/python project/test.csv')


print("Train data shape:",train.shape)
print("Test data shape:",test.shape)

print(train.head())

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)

# to get more information like count, mean, std, min, max etc
print (train.SalePrice.describe())

# to plot a histogram of SalePrice
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

# use np.log() to transform train.SalePric and calculate the skewness a second time, as well as re-plot the data
target = np.log(train.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

# return a subset of columns matching the specified data types
numeric_features = train.select_dtypes(include=[np.number])
# numeric_features.dtypes
print(numeric_features.dtypes)

# displays the correlation between the columns and examine the correlations between the features and the target.
corr = numeric_features.corr()

# The first five features are the most positively correlated with SalePrice, while the next five are the most negatively correlated.
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

#to get the unique values that a particular column has.
#train.OverallQual.unique()
print(train.OverallQual.unique())

#investigate the relationship between OverallQual and SalePrice.
#We set index='OverallQual' and values='SalePrice'. We chose to look at the median here.
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot)

#visualize this pivot table more easily, we can create a bar plot
#Notice that the median sales price strictly increases as Overall Quality increases.
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#to generate some scatter plots and visualize the relationship between the Ground Living Area(GrLivArea) and SalePrice
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

# do the same for GarageArea.
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

# create a new dataframe with some outliers removed
train = train[train['GarageArea'] < 1200]

# display the previous graph again without outliers
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)     # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

# create a DataFrame to view the top null columns and return the counts of the null values in each column
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
#nulls
print(nulls)

#to return a list of the unique values
print("Unique values are:", train.MiscFeature.unique())

# consider the non-numeric features and display details of columns
categoricals = train.select_dtypes(exclude=[np.number])
#categoricals.describe()
print(categoricals.describe())

# When transforming features, it's important to remember that any transformations that you've applied to the training data before
# fitting the model must be applied to the test data.

#Eg:
print ("Original: \n")
print (train.Street.value_counts(), "\n")

# our model needs numerical data, so we will use one-hot encoding to transform the data into a Boolean column.
# create a new column called enc_street. The pd.get_dummies() method will handle this for us
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

print ('Encoded: \n')
print (train.enc_street.value_counts())  # Pave and Grvl values converted into 1 and 0

# look at SaleCondition by constructing and plotting a pivot table, as we did above for OverallQual
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# encode this SaleCondition as a new feature by using a similar method that we used for Street above
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

# explore this newly modified feature as a plot.
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#   Dealing with missing values                                                                     
#   We'll fill the missing values with an average value and then assign the results to data         
#   This is a method of interpolation
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# Check if the all of the columns have 0 null values.
print(sum(data.isnull().sum() != 0))

# separate the features and the target variable for modeling.
# We will assign the features to X and the target variable(Sales Price)to y.

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

#   If we train the model on all of the test data, it will be difficult to tell if overfitting has taken place.
# also state how many percentage from train data set, we want to take as test data set
# In this example, about 33% of the data is devoted to the hold-out set.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


# First, we instantiate the model.
lr = linear_model.LinearRegression()

# ---- fit the model / Model fitting
# lr.fit() method will fit the linear regression on the features and target variable that we pass.
model = lr.fit(X_train, y_train)

# ---- Evaluate the performance and visualize results
# r-squared value is a measure of how close the data are to the fitted regression line
# a higher r-squared value means a better fit(very close to value 1)
print("R^2 is: \n", model.score(X_test, y_test))

# use the model we have built to make predictions on the test data set.
predictions = model.predict(X_test)

# calculates the rmse
print('RMSE is: \n', mean_squared_error(y_test, predictions))


# view this relationship between predictions and actual_values graphically with a scatter plot.
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

# experiment by looping through a few different values of alpha, and see how this changes our results.

for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()

    # if you examined the plots you can see these models perform almost identically to the first model.
    # In our case, adjusting the alpha did not substantially improve our model.

    print("R^2 is: \n", model.score(X_test, y_test))

    # create a csv that contains the predicted SalePrice for each observation in the test.csv dataset.
    submission = pd.DataFrame()
    # The first column must the contain the ID from the test data.
    submission['Id'] = test.Id

    # select the features from the test data for the model as we did above.
    feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()

    # generate predictions
    predictions = model.predict(feats)

    # transform the predictions to the correct form
    # apply np.exp() to our predictions becasuse we have taken the logarithm(np.log()) previously.
    final_predictions = np.exp(predictions)

    print("28 \n")

    # check the difference
    print("Original predictions are: \n", predictions[:10], "\n")
    print("Final predictions are: \n", final_predictions[:10])

    print("29 \n")
    # assign these predictions and check
    submission['SalePrice'] = final_predictions
    # submission.head()
    print(submission.head())

    # export to a .csv file as Kaggle expects.
    # pass index=False because Pandas otherwise would create a new index for us.
    submission.to_csv('submission1.csv', index=False)


    print("\n Finish")


