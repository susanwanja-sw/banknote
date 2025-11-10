# %% [markdown]
# # Importing dependencies

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% [markdown]
# # Data Collection and Processing

# %%
# loading the data
banknote_authentification = pd.read_csv(r"D:\Machine learning projects\me'\bank\data_banknote_authentication.csv", header=None)


# %%
# first few rows
banknote_authentification.head()

# %%
# number of rows and columns
banknote_authentification.shape

# %%
# statistical summary
banknote_authentification.describe()

# %%
# value count - shows how many times each unique value (0--> not fake & 1-->fake) appears in a column
banknote_authentification[4].value_counts()

# %%
# separating the data labels
x = banknote_authentification.drop(columns=4, axis=1)
y= banknote_authentification[4]
print(x)
print(y)

# %% [markdown]
# # Model & Testing Data

# %%
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=1)

# %%
print(x.shape, x_train.shape, x_test.shape)

# %% [markdown]
# # Model

# %%
model = LogisticRegression()

# %%
# training the Logistic Regression with training data
model.fit(x_train, y_train)

# %% [markdown]
# # Model Evaluation

# %%
# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('accuracy on training data:', training_data_accuracy)

# %%
# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('accuracy on test data:', test_data_accuracy)

# %% [markdown]
# # Predictive system

# %%
input_data = [[1.3684,9.6718,-3.9606,-3.1625],[-0.28015,3.0729,-3.3857,-2.9155],[2.3456,0.7888, 0.9088, -0.9877]]
# change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

prediction = model.predict(input_data_as_numpy_array)
print(prediction)

for i, result in enumerate(prediction):
    if result == 0:
        print(f"Banknote {i+1}: REAL (0)")
    else:
        print(f"Banknote {i+1}: FAKE (1)")

# %%
import pickle


# %%
# Save the trained model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

print("Model saved as model.pkl")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



