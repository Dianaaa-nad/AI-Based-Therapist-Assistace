from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import subprocess
from openai import OpenAI
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

st.set_page_config(layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
     tabs = on_hover_tabs(tabName=['About', 'Chat', 'Beck Depression Index'], 
                             iconName=['dashboard', 'chat', 'book'],
                             styles = {'navtab': {'background-color':'#4CA1AF',
                                                  'color': '#2C3E50',
                                                  'font-size': '18px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                             key="1")

if tabs == 'About':
    st.title("About this Website")
    st.markdown(
        """
        This website is a result of groundbreaking research in AI Therapist Assistance, focusing on depression detection and suicide prevention. Our platform features a chatbot and utilizes the Beck Depression Index for depression detection.

        **Key Features:**
        - 24/7 Accessibility: Our system is available at any time, ensuring timely interventions and support.
        - Chatbot: Engage with our AI-powered chatbot for personalized assistance and emotional support.
        - Depression Detection: We employ the Beck Depression Index to identify and assess depression levels.
        - Proactive Monitoring: Caregivers and therapists can monitor individuals at risk and take proactive measures.

        _Empowering Lives, One Conversation at a Time._

        
        """
    )

    # Add some additional styling
    st.markdown(
        """
        <style>
            .css-17a9c07 {
                background-color: #f0f0f0;  /* Light gray background */
                padding: 20px;
                border-radius: 10px;
            }
            .css-1l2xetg {
                color: #2f4f4f;  /* Dark slate gray text color */
            }
            .css-1qkll73 {
                font-size: 20px;  /* Increase font size */
            }
            img {
                max-width: 100%;  /* Ensure images don't overflow */
                border-radius: 10px;  /* Add rounded corners to images */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif tabs == 'Chat':
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
            return "That's really hurt to losing someone that was meant to us, can you descrive what you're feeling right now?"
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


elif tabs == 'Beck Depression Index':

    def bdi_quiz():
        st.title("Beck Depression Inventory (BDI) Quiz")

        # BDI Questions
        questions = [
            {"question": "Do you feel sad?", "options": ["No I don't", "Yes, I feel sad", "I am sad all the time and I can't snap out of it.", "I am so sad and unhappy that I can't stand it."], "score_mapping": [0, 1, 2, 3]},
            {"question": "What do you feel about your future?", "options": ["I am not particularly discouraged about the future. ", "I feel discouraged about the future.", "I feel I have nothing to look forward to.", "I feel the future is hopeless and that things cannot improve. "], "score_mapping": [0, 1, 2, 3]},
            {"question": "Have you ever feel yourself like a failure?", "options": ["No I dont", "I feel I have failed more than the average person", "As I look back on my life, all I can see is a lot of failures", "I feel I am a complete failure as a person."], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do ever you feel guilty?", "options": ["I don't feel particularly guilty", "I feel guilty a good part of the time", "I feel quite guilty most of the time", "I feel guilty all of the time"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel like you have disappointed your family or friends?", "options": ["I don't feel like I have let people down", "I feel I have disappointed my family or friends", "I feel I have let people down", "I feel I am a great disappointment to my family or friends"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you criticize or blame yourself for things?", "options": ["I don't criticize or blame myself more than usual", "I am more self-blaming than I usually am", "I criticize myself for all of my faults", "I blame myself for everything bad that happens"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Have you lost interest in other people?", "options": ["I have not lost interest in other people", "I am less interested in other people than I used to be", "I have lost most of my interest in other people", "I have lost all interest in other people"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel you have no one to talk to?", "options": ["I can talk to someone if I want to", "There are people I can talk to, but I don't want to", "I don't have anyone to talk to", "I feel there is no one I can talk to"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you have trouble making decisions?", "options": ["I can make decisions as well as ever", "I avoid making decisions more than I used to", "I have greater difficulty in making decisions than I used to", "I can't make decisions at all anymore"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel your life is empty?", "options": ["My life is pretty full", "I don't feel much pleasure in things I used to", "I feel I have nothing to look forward to", "I feel my life is empty"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel hopeless about the future?", "options": ["I don't feel hopeless about the future", "I feel somewhat more hopeful about the future", "I feel the future is hopeless and that things cannot improve", "I feel the future is hopeless and that things will not improve"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Have you lost interest in sex?", "options": ["I have not noticed any recent change in my interest in sex", "I am less interested in sex than I used to be", "I have almost no interest in sex", "I have lost interest in sex completely"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel restless and agitated?", "options": ["I don't feel more restless or agitated than usual", "I feel more restless or agitated than usual", "I feel so restless or agitated that I find it hard to stay still", "I am so restless or agitated that I have to keep moving or doing something"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel tired all the time?", "options": ["I am no more tired than usual", "I get tired more easily than I used to", "I am too tired to do most of the things I used to do", "I am too tired to do anything"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you have difficulty concentrating?", "options": ["I can concentrate as well as ever", "I can't concentrate as well as usual", "It's hard to keep my mind on anything for very long", "I find I can't concentrate on anything"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you have trouble sleeping?", "options": ["I can sleep as well as usual", "I don't sleep as well as I used to", "I wake up 1-2 hours earlier than usual and find it hard to get back to sleep", "I wake up several hours earlier than I used to and cannot get back to sleep"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you lose or gain weight unintentionally?", "options": ["I have not experienced any change in my weight", "I feel I have lost weight", "I feel I have gained weight", "I have lost more than 5 pounds or gained more than 5 pounds"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you feel you are a bad person?", "options": ["I don't feel I am a worse person than usual", "I feel I am not as good a person as I should be", "I feel I am a complete failure as a person", "I feel I am a bad person"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you think about suicide?", "options": ["I don't have any thoughts of killing myself", "I have thoughts of killing myself, but I would not carry them out", "I would like to kill myself", "I would kill myself if I had the chance"], "score_mapping": [0, 1, 2, 3]},
            {"question": "Do you cry more than usual?", "options": ["I don't cry any more than usual", "I cry more than I used to", "I cry over every little thing", "I feel like crying, but I can't"], "score_mapping": [0, 1, 2, 3]}
        ]

        # Initialize user's total score
        user_score = 0

        # Display and process BDI questions
        for i, question_data in enumerate(questions, start=1):
            question_text = question_data["question"]
            options = question_data["options"]
            score_mapping = question_data["score_mapping"]

            # Display the question
            st.write(f"Q{i}: {question_text}")

            # User selects an answer
            user_response = st.radio("Select your answer:", options)

            # Map user response to numerical values for scoring
            response_mapping = dict(zip(options, score_mapping))
            user_score += response_mapping[user_response]

            # Display the user's response
            st.write(f"Your response: {user_response}")

        # Display user's total score
        st.write(f"Your total BDI score is: {user_score}")

        # Provide interpretation of the BDI score
        if user_score <= 13:
            result = "You have minimal or no depression."
        elif 14 <= user_score <= 19:
            result = "You have mild depression."
        elif 20 <= user_score <= 28:
            result = "You have moderate depression."
        else:
            result = "You have severe depression."

        st.write(result)

    # Run the BDI quiz
    bdi_quiz()