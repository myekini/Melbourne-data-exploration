import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# load data and preview
melbourne_file_path = "melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.describe()

#set the prediction target
melbourne_data.columns
y = melbourne_data.Price

# features to be consider for prediction
melbourne_features = ['Rooms', 'Bathroom','Landsize','Lattitude','Longtitude']
X = melbourne_data[melbourne_features]
X.head()


#define model and fit (capture patterns)
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X,y)

#predict first few data
print('making predictions for the following five houses')
print(X.head())
print("The predictions are...")
print(melbourne_model.predict(X.head()))

# trying to measure the accuracy of our model the less smart way
y.head()

# evaluating the quality of the model using Mean Absolute Error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model using a decision tree
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
print(val_predictions)


# Define model using a RandomForestRegressor
melbourne_model_2 = RandomForestRegressor(random_state=1)
melbourne_model_2.fit(train_X, train_y)

#predict the house prices on the validation data
val_predictions_2 = melbourne_model_2.predict(val_X)
print(mean_absolute_error(val_y, val_predictions_2))


# define a function that builds a model and makes predictions
def Scoreall(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=10,random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y,preds)
Scoreall(train_X, val_X, train_y, val_y)


# get columns with missing data
cols_with_missing = [col for col in train_X.columns
                    if train_X[col].isnull().any()]

#drop columns in training and validation data
reduced_X_train = train_X.drop(cols_with_missing,axis=1)
reduced_X_val = val_X.drop(cols_with_missing,axis=1)

print("MAE from approache 1(drop columns with missing data)")
print(Scoreall(reduced_X_train, reduced_X_val, train_y, val_y))

# build a pipeline 
#preprocessing for missing numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#columns with numerical data
numerical_cols = [col for col in train_X.columns
                 if train_X[col].dtype in ['int64', 'float64']]

#columns with categorical data
categorical_cols = [col for col in train_X.columns
                   if train_X[col].nunique() < 10 and 
                   train_X[col].dtype == "object"]

#Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


#define the model. we'd be using randomForestRegressor
model = RandomForestRegressor(n_estimators=100 , random_state=0)

#bundle preprocessing and modelling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)])

#preprocessing of training data fit model
my_pipeline.fit(train_X, train_y)

#preprocessing of validation data and get predictions
preds = my_pipeline.predict(val_X)

#Evaluate the model
score = mean_absolute_error(val_y,preds)
print('MAE:', score)


# save output to a csv file
output = pd.DataFrame({'Id':val_X.index, 'SalePrice': val_predictions_2})
output.to_csv('melbourne.csv', index=False)