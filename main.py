from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# data file path
train_file = './data/home-data-fo-ml-course/train.csv'
test_file = './data/home-data-fo-ml-course/test.csv'

# read data file
train = pd.read_csv(train_file, index_col='Id')
test = pd.read_csv(test_file, index_col='Id')

# remove rows with missing target
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

# separate y from x
y_train = train.SalePrice
x_train = train.drop(['SalePrice'], axis=1)

categorical_cols = [cname for cname in x_train.columns
                    if x_train[cname].nunique() < 10 and
                    x_train[cname].dtype == "object"]

# get numerical columns
numerical_cols = [cname for cname in x_train.columns if
                  x_train[cname].dtype in ['int64', 'float64']]

# preprocessing for numerical data
numerical_transformer = SimpleImputer()

#preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor()
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                      ])

clf.fit(x_train, y_train)

preds = clf.predict(test)

#### output model results
output = pd.DataFrame({
    'Id':test.index,
    'SalePrice': preds})

output.to_csv('submission.csv', index = False)