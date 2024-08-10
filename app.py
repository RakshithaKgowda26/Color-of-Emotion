# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Your existing code here
# from pymongo import MongoClient
# client=MongoClient("mongodb+srv://sahanashivakumar2002:YVFSadXO97vX2W4G@cluster0.kuekten.mongodb.net/")
# db=client['Mini_project']
# collection=db["Train_set"]

# import pandas as pd
# doc=collection.find()
# df=pd.DataFrame(doc)

# # Extract features and labels
# train_sentences = df['Text'].tolist()
# train_emotions = df['Emotion'].tolist()
# train_colors = df['Color'].tolist()

# from sklearn.feature_extraction.text import TfidfVectorizer
# # Use TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(train_sentences)

# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Split the data into training and validation sets
# X_train, X_val, y_emotion_train, y_emotion_val, y_color_train, y_color_val = train_test_split(
#     X_train, train_emotions, train_colors, test_size=0.2, random_state=42
# )

# # Train a classifier for emotions
# emotion_classifier = SVC(kernel='linear')
# emotion_classifier.fit(X_train, y_emotion_train)

# # Train a classifier for colors
# color_classifier = SVC(kernel='linear')
# color_classifier.fit(X_train, y_color_train)

# # Validate the models
# emotion_predictions = emotion_classifier.predict(X_val)
# color_predictions = color_classifier.predict(X_val)

# # Print accuracy
# #print("Emotion Accuracy:", accuracy_score(y_emotion_val, emotion_predictions))
# #print("Color Accuracy:", accuracy_score(y_color_val, color_predictions))


# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         new_sentence = request.form['sentence']

#         # Vectorize the new sentence
#         X_new = vectorizer.transform([new_sentence])

#         # Predict emotion and color
#         predicted_emotion = emotion_classifier.predict(X_new)[0]
#         predicted_color = color_classifier.predict(X_new)[0]

#         return render_template('index.html', predicted_emotion=predicted_emotion, predicted_color=predicted_color, sentence=new_sentence)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request

app = Flask(__name__)

# Your existing code here
from pymongo import MongoClient
client=MongoClient("mongodb+srv://sahanashivakumar2002:YVFSadXO97vX2W4G@cluster0.kuekten.mongodb.net/")
db=client['Mini_project']
collection=db["Train_set"]

import pandas as pd
doc=collection.find()
df=pd.DataFrame(doc)

# Extract features and labels
train_sentences = df['Text'].tolist()
train_emotions = df['Emotion'].tolist()
train_colors = df['Color'].tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
# Use TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split the data into training and validation sets
X_train, X_val, y_emotion_train, y_emotion_val, y_color_train, y_color_val = train_test_split(
    X_train, train_emotions, train_colors, test_size=0.2, random_state=42
)

# Train a classifier for emotions
emotion_classifier = SVC(kernel='linear')
emotion_classifier.fit(X_train, y_emotion_train)

# Train a classifier for colors
color_classifier = SVC(kernel='linear')
color_classifier.fit(X_train, y_color_train)

# Validate the models
emotion_predictions = emotion_classifier.predict(X_val)
color_predictions = color_classifier.predict(X_val)

# Print accuracy
print("Emotion Accuracy:", accuracy_score(y_emotion_val, emotion_predictions))
print("Color Accuracy:", accuracy_score(y_color_val, color_predictions))


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_sentence = request.form['sentence']

        # Vectorize the new sentence
        X_new = vectorizer.transform([new_sentence])

        # Predict emotion and color
        predicted_emotion = emotion_classifier.predict(X_new)[0]
        predicted_color = color_classifier.predict(X_new)[0]

        return render_template('index.html', predicted_emotion=predicted_emotion, predicted_color=predicted_color, sentence=new_sentence)

if __name__ == '__main__':
    app.run(debug=True)