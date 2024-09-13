# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example: Loading dataset
data = pd.read_csv('C:/Users/ADMIN/Documents/model_deployment/kalki_movie_reviews.csv')

# Preprocessing: Example for text data
X = data['Comments']
y = data['Ratings']

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')