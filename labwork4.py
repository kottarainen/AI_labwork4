import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam

# Load the dataset
dataset = pd.read_csv("full_dataset.csv")

# Explore the dataset
print("First 5 rows of the dataset:")
print(dataset.head())

print("\nDataset info:")
print(dataset.info())

print("\nDescriptive statistics of the dataset:")
print(dataset.describe())
dataset.dropna(inplace=True)

# Encoding categorical variables
categorical_cols = ['Suburb', 'Type', 'Regionname', 'CouncilArea']
encoder = OneHotEncoder()
encoded_data = pd.DataFrame(encoder.fit_transform(dataset[categorical_cols]).toarray(), columns=encoder.get_feature_names_out(categorical_cols))
encoded_dataset = pd.concat([dataset.drop(columns=categorical_cols), encoded_data], axis=1)

# Splitting dataset into features and target variable
X = encoded_dataset.drop(columns=["Price", "Address", "Method", "SellerG", "Date", "Postcode", "Bedroom2", "YearBuilt", "Lattitude", "Longtitude", "Propertycount"])
y = encoded_dataset["Price"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Handle missing values in the target variable
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

#RNN model
# Reshape the input data to match the expected shape of the model
X_train_rnn = X_train_imputed.reshape((X_train_imputed.shape[0], 1, X_train_imputed.shape[1]))
X_test_rnn = X_test_imputed.reshape((X_test_imputed.shape[0], 1, X_test_imputed.shape[1]))
num_samples = min(X_train_rnn.shape[0], y_train.shape[0])
X_train_rnn = X_train_rnn[:num_samples]
y_train = y_train[:num_samples]
# Define and train the RNN model
model_rnn = Sequential([
    SimpleRNN(units=32, activation='relu', input_shape=(1, X_train_rnn.shape[2])),
    Dense(units=1)
])
model_rnn.compile(optimizer=Adam(), loss='mean_squared_error')
history = model_rnn.fit(X_train_rnn, y_train, epochs=20, batch_size=32, validation_split=0.2)
# RNN model evaluation
y_pred_rnn = model_rnn.predict(X_test_rnn).flatten()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Linear regression
num_samples = min(X_train_imputed.shape[0], y_train.shape[0])
X_train_imputed = X_train_imputed[:num_samples]
y_train = y_train[:num_samples]
linear_regression = LinearRegression()
# Fit model to training data
linear_regression.fit(X_train_imputed,y_train)
# Predict
y_pred_lr = linear_regression.predict(X_test_imputed)

# Polynomial regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_imputed)
X_test_poly = poly_features.transform(X_test_imputed)

# Feature scaling for polynomial features
#poly_scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)
# Ridge Polynomial Regression with regularization
#ridge_poly_regression = Ridge(alpha=1.0)
#ridge_poly_regression.fit(X_train_poly, y_train)
#y_pred_ridge_pr = ridge_poly_regression.predict(X_test_poly)

# Fit the polynomial regression model
poly_regression = LinearRegression()
poly_regression.fit(X_train_poly_scaled, y_train)

# Make predictions
y_pred_poly = poly_regression.predict(X_test_poly_scaled)

# Random Forest
random_forest = RandomForestRegressor(n_estimators=200, random_state=95)
random_forest.fit(X_train_imputed, y_train)
y_pred_rf = random_forest.predict(X_test_imputed)

# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=95)
decision_tree.fit(X_train_imputed, y_train)
y_pred_dt = decision_tree.predict(X_test_imputed)

models = {"Linear Regression": y_pred_lr, "Polynomial Regression": y_pred_poly, "Random Forest": y_pred_rf, "RNN": y_pred_rnn, "Decision Tree": y_pred_dt}

# Evaluation metrics
evaluation_metrics = {}
for name, prediction in models.items():
    prediction = prediction[:len(y_test)] #slice prediction array to match the length of y_test
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    evaluation_metrics[name] = {'MAE': mae, 'MSE': mse}
    
for name, metrics in evaluation_metrics.items():
    print(f"{name} Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

