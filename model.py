import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("winequality.csv")

print(df.head())

# Select independent and dependent variable
x = df.drop("quality", axis = True)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model = RandomForestClassifier(max_depth = 5, n_estimators = 10).fit(X_train.values,y_train)
y_pred = model.predict(X_test)
# Split the dataset into train and test




# # Instantiate the model
# model = RandomForestRegressor()

# # Fit the model
# model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(model, open("winequality", "wb"))