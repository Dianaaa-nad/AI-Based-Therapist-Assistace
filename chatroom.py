from openai import OpenAI
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
import text_hammer as th
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import joblib
import re

def remove_emails(text):
    # Manual implementation to remove emails
    return re.sub(r'\S+@\S+', '', text)

def remove_html_tags(text):
    # Manual implementation to remove HTML tags
    return re.sub(r'<.*?>', '', text)

def remove_special_chars(text):
    # Manual implementation to remove special characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_accented_chars(text):
    # Manual implementation to remove accented characters
    # You can use unidecode or other methods for a more comprehensive solution
    return re.sub(r'[^\x00-\x7F]+', '', text)


def text_preprocessing(df, col_name):
    column = col_name
    df[column] = df[column].apply(lambda x: str(x).lower())
    df[column] = df[column].apply(lambda x: remove_emails(x))
    df[column] = df[column].apply(lambda x: remove_html_tags(x))
    df[column] = df[column].apply(lambda x: remove_special_chars(x))
    df[column] = df[column].apply(lambda x: remove_accented_chars(x))
    return df

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

def tokenize_columns(df, text_column):
    df[text_column] = df[text_column].apply(lambda x: custom_tokenizer(str(x).lower()))
    return df

def calculate_depression_score(messages):
    depression_keywords = ["sad", "unhappy", "worthless", "empty", "lonely", "discouraged", "suicide", "kill", "failure", "dissatisfied", "guilty", "punished", "disappointed", "disgusted", "hate", "worse", "irritated", "tired", "lost"]

    user_input_text = " ".join([message["content"].lower() for message in messages if message["role"] == "user"])

    depression_score = sum(user_input_text.count(keyword) for keyword in depression_keywords)

    return depression_score

# Load the saved Complement Naive Bayes model
loaded_model = joblib.load('complement_naive_bayes_model.joblib')

# Load the saved CountVectorizer
loaded_vectorizer = joblib.load('count_vectorizer.joblib')

def predict_sentiment(user_query):
    # Preprocess the user query
    user_query_df = pd.DataFrame({'text': [user_query]})
    user_query_df = text_preprocessing(user_query_df, 'text')

    # Tokenize the user query
    user_query_df = tokenize_columns(user_query_df, 'text')

    # Join the tokenized words into a sentence (document)
    user_query_document = user_query_df['text'].apply(lambda x: ' '.join(x))

    # Use the loaded vectorizer to convert the user query into a bag-of-words representation
    X_user_query = loaded_vectorizer.transform(user_query_document)

    # Predict the sentiment using the loaded Complement Naive Bayes model
    prediction = loaded_model.predict(X_user_query)[0]

    # Map the numeric prediction back to class label
    predicted_class = {0: 'non-suicide', 1: 'suicide'}[prediction]

    return predicted_class

def generate_openai_response(prompt, model):
    full_response = ""
    for response in client.chat.completions.create(
        model=model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    ):
        full_response += (response.choices[0].delta.content or "")
    return full_response

def generate_response(user_input):
    full_response = ""
    
    return full_response




def get_keyword_response1(question):
    question_lower = question.lower()
    if "hai" in question_lower:
        return "Hello, how are you?"
    elif "fine, how about you?" in question_lower:
        return "I'm fine as always! I will be here for you anytime you need"
    elif "yes" in question_lower:
        return "You doing great controlling your emotions! Now imagine you are talking with them, what you want to say?"
    elif "job" in question_lower:
        return "That must be hard for you for experiencing something like that."
    elif "girlfriend" in question_lower or "boyfriend" in question_lower:
        return "That's so heartbreaking"
    elif "friends" in question_lower:
        return "I appreciate you opening up about your friends. It sounds like talking about your friends brings up some strong feelings. Can you describe what you're feeling right now?"
    elif "sad" in question_lower:
        return "It's normal to have a range of emotions in relationships but that must be hard for you right now. \nI understand how painfull it can be to feel leave behind, let's take a deep breath and relase it, do it until you heart feel a bit relax. Did you feel better now?"
    elif "angry" in question_lower or "upset" in question_lower:
        return "I appreciate you opening up about your friends. It's normal to have a range of emotions in relationships."
    elif "misunderstood" in question_lower:
        return "I'm so sorry to hear that. Have you tried to communicate with them about this?"
    elif "No, I don't have enough courage to do it" in question_lower:
        return "I'm so sorry to hear that. Have you tried to communicate with them about this?"
    elif "jealous" in question_lower or "insecure" in question_lower:
        return "Feeling jealous with someone that you love is normal. Have you tried to communicate with them about this?"
    else:
        return generate_openai_response(question, st.session_state["openai_model"])

def main():
    st.title("Chat with AI")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "initial_sentiment" not in st.session_state:
        st.session_state.initial_sentiment = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            predicted_sentiment = predict_sentiment(user_input)

            if st.session_state.initial_sentiment is None:
                st.session_state.initial_sentiment = predicted_sentiment

            if predicted_sentiment == 'suicide':
                response = "I'm really sorry to hear that you're feeling this way. Would you like to talk more about it? I'm here for you."
            elif st.session_state.initial_sentiment == 'suicide':
                response = get_keyword_response1(user_input)
            else:
                if user_input == 'Hai':
                    response = "Hello, how are you?"
                else:
                    response = generate_openai_response(user_input, st.session_state["openai_model"])

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            depression_score = calculate_depression_score(st.session_state.messages)
            st.write(f"Depression Score: {depression_score}")
                        

if __name__ == "__main__":
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    main()
