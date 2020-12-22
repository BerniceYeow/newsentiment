# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd

import nltk
from nltk.tokenize import WordPunctTokenizer

import malaya








    

#LOAD TOKENISER, STOPWORDS, MALAYA DEEP MODEL FOR LANGUAGE DETECTION, SPELLER CORRECTOR, NORMALIZE FUNCTION AND ANNOTATION
deep = malaya.language_detection.deep_model()
#fast_text = malaya.language_detection.fasttext()
token = WordPunctTokenizer()
#from nltk.corpus import stopwords
#english_stopwords = stopwords.words('english')
corrector = malaya.spell.probability()



from nltk.tokenize import WordPunctTokenizer

import malaya




# NLTK Stop words

from nltk import word_tokenize
import streamlit as st


from PIL import Image




from nltk.tokenize import WordPunctTokenizer

import malaya









normalize = ['hashtag', 'url', 'email', 'user', 'money', 'time', 'date',
             'duration', 'temperature', 'rest_emoticons']

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("HATI.AI")
    image = Image.open('macroview.jpg')
    #st.image(image, use_column_width=False)
    st.sidebar.image(image)
    st.sidebar.title("Hati.Ai Web App")
    
    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)


    if choice == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
			# if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))
                def process_text(text):
                    processed_data = []
                    # Make all the strings lowercase and remove non alphabetic characters
                    #text = re.sub('[^A-Za-z]', ' ', text.lower())
                
                    # Tokenize the text; this is, separate every sentence into a list of words
                    # Since the text is already split into sentences you don't have to call sent_tokenize
                    tokenized_text = word_tokenize(text)
                
                    #append the result into a new list called processed_data
                    processed_data.append(tokenized_text)
                
                
                    # Remember, this final output is a list of words
                    return processed_data
            
                @st.cache(suppress_st_warning=True)
                def load_data(uploaded_file):
                    
            
                    df = pd.read_csv(uploaded_file)
                            
             
                    return df
                
                st.sidebar.subheader("Choose What Do You Want To Do")
                classifier = st.sidebar.selectbox(" ", ("Find new topics automatically", "POWER BI Dashboard", "Interact with our chatbot"))
                if classifier == 'POWER BI Dashboard':
                    import streamlit.components.v1 as components
                    from urllib.request import urlopen
                    html = urlopen("https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f").read()
                    #components.html(html, width=None, height=600, scrolling=True)
                    st.markdown("""
                        <iframe width="900" height="606" src="https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f" frameborder="0" style="border:0" allowfullscreen></iframe>
                        """, unsafe_allow_html=True)

              
                if classifier == 'Interact with our chatbot':    
                    import pickle
                    with open('tnb_topic_classifier_svm', 'rb') as training_model:
                        topic_model = pickle.load(training_model)
                    import malaya
                    model = malaya.sentiment.transformer(model = 'albert', size = 'base')
                    #from src import model          
                    #malay_bert = model.BertModel()
                    # eng_flair = model.Flair()
                    # eng_vader = model.Vader()
                    test = pd.DataFrame()
                    test['Positive'] = ''
                    test['Neutral'] = ''
                    test['Negative'] = ''
                    
                    st.title("Sentiment Analyzer")
                    message = st.text_area("Enter Text","Type Here ..")
                    if st.button("Analyze"):
                     with st.spinner("Analyzing the text …"):
                         result = model.predict_proba([message])
                         #result = malay_bert.predict(message)
                         message = [message]
                         topic = topic_model.predict(message)
                         #output = "Result is: Positive:" + str(result[0]) + "Neutral:" + str(result[1]) + "Negative:" + str(result[2]) + "topic is: " + str(topic)
                         output = "result is:" + str(result) + "topic is: " + str(topic)
                         st.write(output)
            
                    else:
                     st.warning("Not sure! Try to add some more words")
    
                from stop_words import get_stop_words
                if classifier == 'Find new topics automatically':
            
                    
                    uploaded_file = st.file_uploader('Upload CSV file to begin', type='csv')
                
                    #if upload then show left bar
                    if uploaded_file is not None:
                        df = load_data(uploaded_file)
                
                
                
                        if st.sidebar.checkbox("Show raw data", False):
                            st.subheader("Uploaded Data Set")
                            st.write(df)
                
                
            
                        st.sidebar.subheader("Text column to analyse")
                        st_ms = st.sidebar.selectbox("Select Text Columns To Analyse", (df.columns.tolist()))
                        

                        df_list = list(df)
     
        
                        import top2vec
                        from top2vec import Top2Vec
                        
                        #INITIALIZE AN EMPTY DATAFRAME, CONVERT THE TEXT INTO STRING AND APPEND INTO THE NEW COLUMN
                        d1 = pd.DataFrame()
                        d1['text'] = ""
                        d1['text'] = df[st_ms]
                        d1['text'] = d1['text'].astype(str)
                        
                
                        #INITIALIZE THE TOP2VEC MODEL AND FIT THE TEXT
                        #model.build_vocab(df_list, update=False)
                        model = Top2Vec(documents=d1['text'], speed="learn", workers=10)
                        
                        topic_sizes, topic_nums = model.get_topic_sizes()
                        for topic in topic_nums:
                            st.pyplot(model.generate_topic_wordcloud(topic))
                            # Display the generated image:

        


            else:
                st.warning("Incorrect Username/Password")


    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")






if __name__ == '__main__':
    main()