import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#fiile path to csv
mushroom_file_path = 'mushrooms.csv'
mushroom_data = pd.read_csv(mushroom_file_path)

#specify targets and prediction variables
y = mushroom_data.Class
predict_features = ['cap-shape', 'cap-surface','cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring' , 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
X = mushroom_data[predict_features]

#split training and testing data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 2)

#preprocess data
preprocess = OneHotEncoder(handle_unknown = 'ignore')

#specify model
mushroom_model = LogisticRegression(random_state = 7)

#bundle preprocessing and modeling into pipeline
pipeLine = Pipeline(steps = [('preprocessor', preprocess), ('model', mushroom_model)])

#fit the pipeline
pipeLine.fit(train_X, train_y)

#make predictions
predictions = pipeLine.predict(test_X)
print(predictions)
print(test_y)