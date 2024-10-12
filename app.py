import re
import streamlit as st
import pickle
import nltk

# Download necessary nltk resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# Define the Streamlit app
st.title("Amazon Musical Instruments Review Sentiment Analysis")


# Get user input
user_input = st.text_input("Enter your review:")


# Preprocess the user input
def preprocess_text(input_text, use_stemming=True, use_lemmatization=True):
    # Convert to lowercase
    preprocessed_text = input_text.lower()

    # Remove punctuation
    preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(preprocessed_text)
    filtered_words = [word for word in word_tokens if word not in stop_words]

    # Initialize stemmer and lemmatizer
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Optionally apply stemming and lemmatization
    if use_stemming:
        filtered_words = [stemmer.stem(word) for word in filtered_words]
    if use_lemmatization:
        filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    preprocessed_text = ' '.join(filtered_words)

    return preprocessed_text


# Make prediction
if user_input:
    # Preprocess the input text
    preprocessed_input = preprocess_text(user_input)

    # Vectorize the preprocessed input
    input_vector = vectorizer.transform([preprocessed_input])

    # Make prediction
    prediction = model.predict(input_vector)[0]

    # Display the prediction
    st.write("Prediction:")
    if prediction == 1:
        st.write("Positive")
    else:  
        st.write("Negative") # Assuming you have binary classification (1 for positive, others for negative)