# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from nltk.corpus import stopwords

import malaya


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sys
import time


import json


import datetime as dt

#LOAD THE WORDS TO REMOVE FROM MALAYA TATABAHASA DICTIONARY
from malaya.text import tatabahasa

words_to_remove = tatabahasa.stopwords
words_to_remove1 = tatabahasa.gantinama_list
words_to_remove2 = tatabahasa.laughing
words_to_remove3 = tatabahasa.tanya_list
words_to_remove4 = tatabahasa.sendi_list

#LOAD SHORT FORMS
malayshortform = json.load(open('Short Form words-Malay (latest).json'))
englishshortform = json.load(open('Shortform - English (New) (1).json'))


#LOAD THE WORDS FROM MALAYA AND VADER CORPUS AND SOME SHORT FORMS MALAY WORDS
corpus1 = pd.read_csv('wordstoremove.csv', encoding = 'ISO-8859-1')

#APPEND THE WORDS FROM MALAYA AND VADER CORPUS AND SOME SHORT FORMS MALAY WORDS INTO ANOTHER LIST
corpus2 = []
for i in corpus1.words:
    corpus2.append(i)
    

#LOAD TOKENISER, STOPWORDS, MALAYA DEEP MODEL FOR LANGUAGE DETECTION, SPELLER CORRECTOR, NORMALIZE FUNCTION AND ANNOTATION
deep = malaya.language_detection.deep_model()
#fast_text = malaya.language_detection.fasttext()
token = WordPunctTokenizer()
#from nltk.corpus import stopwords
#english_stopwords = stopwords.words('english')
corrector = malaya.spell.probability()

import pandas as pd


from nltk.tokenize import WordPunctTokenizer

import malaya


import json


import pandas as pd


# !{sys.executable} -m spacy download en

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words

from nltk import word_tokenize
import streamlit as st
import pandas as pd

from PIL import Image



import pandas as pd


from nltk.tokenize import WordPunctTokenizer

import malaya

import pandas as pd

import json



import pandas as pd






english_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
english_stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know','go', 'get', 'do', 'done', 'try', 'many', 'some', 'think', 'see', 'rather',  'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


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
                classifier = st.sidebar.selectbox(" ", ("Find new topics", "POWER BI Dashboard", "Interact with our chatbot"))
                if classifier == 'POWER BI Dashboard':
                    import streamlit.components.v1 as components
                    from urllib.request import urlopen
                    html = urlopen("https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f").read()
                    #components.html(html, width=None, height=600, scrolling=True)
                    st.markdown("""
                        <iframe width="900" height="606" src="https://app.powerbi.com/view?r=eyJrIjoiZTA4NWU4MjYtOTk3Yi00N2ZhLTgwZWQtZWFhMzNkNDk1Zjk3IiwidCI6Ijk5NmQwYTI3LWUwOGQtNDU1Ny05OWJlLTY3ZmQ2Yjk3OTA0NCIsImMiOjEwfQ%3D%3D&pageName=ReportSection06db5928b6af61b2868f" frameborder="0" style="border:0" allowfullscreen></iframe>
                        """, unsafe_allow_html=True)
                from stop_words import get_stop_words
                if classifier == 'Find new topics':
            
                    
                    uploaded_file = st.file_uploader('Upload CSV file to begin', type='csv')
                
                    #if upload then show left bar
                    if uploaded_file is not None:
                        df = load_data(uploaded_file)
                
                
                
                        if st.sidebar.checkbox("Show raw data", False):
                            st.subheader("Uploaded Data Set")
                            st.write(df)
                
                
            
                        st.sidebar.subheader("Text column to analyse")
                        st_ms = st.sidebar.selectbox("Select Text Columns To Analyse", (df.columns.tolist()))
                        
                        st.sidebar.subheader("Number of Topics")
                        #choose parameters
                        num_topics = st.sidebar.number_input("Enter number of topics", 1, 20, step=1)
                        df_list = list(df)
                        #from stop_words import get_stop_words
                        malay_stop_words = get_stop_words('indonesian')
                        if st.sidebar.button("Classify", key='classify'):
                            def sent_to_words(sentences):
                                for sent in sentences:
                                    sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
                                    sent = re.sub('\s+', ' ', sent)  # remove newline chars
                                    sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
                                    urlPattern        =  r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*|[^ ]*(.com))"
                                    userPattern       = '@[^\s]+'
                                    alphaPattern      = "[^a-zA-Z]"
                                    sequencePattern   = r"(.)\1\1+"
                                    seqReplacePattern = r"\1\1"
                                    
            
                                    
                                    #Convert all letters to lowercase
                                    sent = ' '.join(sent).lower()
                                    
                                    
                                    # Replace all URls with 'URL'
                                    sent = re.sub(urlPattern,' ',str(sent))
                                        
                                    # Replace @USERNAME to 'USER'.
                                    sent = re.sub(userPattern,' ', str(sent)) 
                                    
                                    # Replace all the emojis and emoticons with their own coding
                                    for emot in UNICODE_EMO:
                                        sent = sent.replace(emot, " " + emot + " ")
                                        sent = sent.replace(emot, " ".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
                                    
                                    for emot in EMOTICONS:
                                        sent = sent.replace(emot, " " + emot + " ")
                                        sent = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), sent)
                                    # Replace all non alphabets.
                                    sent = re.sub(alphaPattern,' ', sent)
                                
                                    # Replace 3 or more consecutive letters by 2 letter.
                                    sent = re.sub(sequencePattern, seqReplacePattern, sent)
                                    
                                    # Tokenize the words
                                    tokens = token.tokenize(sent)
                                
                                    # Remove stopwords
                                    words = [w for w in tokens if not w in english_stopwords]

                                    words = [w for w in words if not w in malay_stop_words]
            
                                
                                    #JOIN BACK THE WORDS
                                    sent = (" ".join(words)).strip()
            
                                    yield (sent)
                            df[st_ms] = df[st_ms].astype(str)
                            data = df[st_ms].values.tolist()
                            data_words = list(sent_to_words(data))
                            
            
                            #data = df[st_ms]
                            #data = data.values.tolist()
                            #data_words = list(data)
                            
                            
                            # Build the bigram and trigram models
                            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
                            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
                            bigram_mod = gensim.models.phrases.Phraser(bigram)
                            trigram_mod = gensim.models.phrases.Phraser(trigram)
                            
                            processed_data = []
                            from gensim import corpora
                            # !python3 -m spacy download en  # run in terminal once
                            def process_words(texts, stop_words=english_stopwords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                                """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
                                texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
                                texts = [bigram_mod[doc] for doc in texts]
                                texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
                                #texts_out = []
                                #nlp = spacy.load('en', disable=['parser', 'ner'])
                                #for sent in texts:
                                    #doc = nlp(" ".join(sent))
                                    #texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                                # remove stopwords once more after lemmatization
                                #texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
                                return texts
            
            
                            data_ready = process_words(data_words)
                            # Create Dictionary
                            id2word = corpora.Dictionary(data_ready)
                            
                            # Create Corpus: Term Document Frequency
                            corpus = [id2word.doc2bow(text) for text in data_ready]
                            
                            # Build LDA model
                            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                                        id2word=id2word,
                                                                        num_topics=num_topics,
                                                                        random_state=42,
                                                                        update_every=1,
                                                                        chunksize=10,
                                                                        passes=10,
                                                                        alpha='symmetric',
                                                                        iterations=100,
                                                                        per_word_topics=True)
                            
                            
                            
                            def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
                                # Init output
                                sent_topics_df = pd.DataFrame()
                            
                                # Get main topic in each document
                                for i, row_list in enumerate(ldamodel[corpus]):
                                    row = row_list[0] if ldamodel.per_word_topics else row_list
                                    # print(row)
                                    row = sorted(row, key=lambda x: (x[1]), reverse=True)
                                    # Get the Dominant topic, Perc Contribution and Keywords for each document
                                    for j, (topic_num, prop_topic) in enumerate(row):
                                        if j == 0:  # => dominant topic
                                            wp = ldamodel.show_topic(topic_num)
                                            topic_keywords = ", ".join([word for word, prop in wp])
                                            sent_topics_df = sent_topics_df.append(
                                                pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                                        else:
                                            break
                                sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
                            
                                # Add original text to the end of the output
                                contents = pd.Series(texts)
                                sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
                                return (sent_topics_df)
                            
                            df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
                            
                            # Format
                            df_dominant_topic = df_topic_sents_keywords.reset_index()
                            df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
            
                            
                            # Display setting to show more characters in column
                            pd.options.display.max_colwidth = 100
                            
                            sent_topics_sorteddf_mallet = pd.DataFrame()
                            sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
                            
                            for i, grp in sent_topics_outdf_grpd:
                                sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                                         grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                                        axis=0)
                            
                            # Reset Index    
                            sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
                            
                            # Format
                            sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
                            
                            # Show
                            sent_topics_sorteddf_mallet.head(10)
                            
                            import seaborn as sns
                            import matplotlib.colors as mcolors
                            from collections import Counter
                            topics = lda_model.show_topics(formatted=False)
                            data_flat = [w for w_list in data_ready for w in w_list]
                            counter = Counter(data_flat)
                            
                            out = []
                            for i, topic in topics:
                                for word, weight in topic:
                                    out.append([word, i , weight, counter[word]])
                            
                            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])  
                        
                    
                    
                    
                    
                            st.subheader("LDA Results")
                            # Plot Word Count and Weights of Topic Keywords
                            fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
                            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
                            for i, ax in enumerate(axes.flatten()):
                                ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
                                ax_twin = ax.twinx()
                                ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
                                ax.set_ylabel('Word Count', color=cols[i])
                                ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
                                ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
                                ax.tick_params(axis='y', left=False)
                                ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
                                ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
                            
                            fig.tight_layout(w_pad=2)    
                            fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
                            st.pyplot(fig)
                            
                            # Sentence Coloring of N Sentences
                            def topics_per_document(model, corpus, start=0, end=1):
                                corpus_sel = corpus[start:end]
                                dominant_topics = []
                                topic_percentages = []
                                for i, corp in enumerate(corpus_sel):
                                    topic_percs, wordid_topics, wordid_phivalues = model[corp]
                                    dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
                                    dominant_topics.append((i, dominant_topic))
                                    topic_percentages.append(topic_percs)
                                return(dominant_topics, topic_percentages)
                            
                            dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            
                            
                            # Distribution of Dominant Topics in Each Document
                            df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
                            dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
                            df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
                            
                            # Total Topic Distribution by actual weight
                            topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
                            df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
                            
                            # Top 3 Keywords for each Topic
                            topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                                             for j, (topic, wt) in enumerate(topics) if j < 3]
                            
                            df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
                            df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
                            df_top3words.reset_index(level=0,inplace=True)
                            
                            from matplotlib.ticker import FuncFormatter
            
                            # Plot
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)
                            
                            # Topic Distribution by Dominant Topics
                            ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
                            ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
                            tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
                            ax1.xaxis.set_major_formatter(tick_formatter)
                            ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
                            ax1.set_ylabel('Number of Documents')
                            ax1.set_ylim(0, 1000)
                            
                            # Topic Distribution by Topic Weights
                            ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
                            ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
                            ax2.xaxis.set_major_formatter(tick_formatter)
                            ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
                            
                            st.pyplot(fig)
                            
                            # Get topic weights and dominant topics ------------
                            from sklearn.manifold import TSNE
                            from bokeh.plotting import figure, output_file, show
                            from bokeh.models import Label
                            from bokeh.io import output_notebook
                            
                            # Get topic weights
                            topic_weights = []
                            for i, row_list in enumerate(lda_model[corpus]):
                                topic_weights.append([w for i, w in row_list[0]])
                            
                            # Array of topic weights    
                            arr = pd.DataFrame(topic_weights).fillna(0).values
                            
                            # Keep the well separated points (optional)
                            arr = arr[np.amax(arr, axis=1) > 0.35]
                            
                            # Dominant topic number in each doc
                            topic_num = np.argmax(arr, axis=1)
                            
                            # tSNE Dimension Reduction
                            tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
                            tsne_lda = tsne_model.fit_transform(arr)
                            
                            # Plot the Topic Clusters using Bokeh
                            output_notebook()
                            n_topics = num_topics
                            mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
                            plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                                          plot_width=900, plot_height=700)
                            plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
            
                            st.write(plot)
                            import streamlit.components.v1 as components
                            import pyLDAvis.gensim
                            pyLDAvis.enable_notebook()
                            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
                            pyLDAvis.save_html(vis, 'lda.html')
                            
                            #html_file = open("c:/Users/BerniceYeow/Documents/Microview Learning/lda.html", "w")
                            from urllib.request import urlopen
                            import os
                            cwd = os.getcwd()
                            #html = urlopen("file:///lda.html").read()
                            html = urlopen("file:///" + cwd + "/lda.html").read()
            
            
                            components.html(html, width=None, height=600, scrolling=True)
                            #st.markdown(vis, unsafe_allow_html=True)
                            
                            #st.iframe(vis)
              
                if classifier == 'Interact with our chatbot':    
                    import pickle
                    with open('tnb_topic_classifier_svm', 'rb') as training_model:
                        topic_model = pickle.load(training_model)
                    from src import model          
                    malay_bert = model.BertModel()
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
                         result = malay_bert.predict(message)
                         message = [message]
                         topic = topic_model.predict(message)
                         output = "Result is: Positive:" + str(result[0]) + "Neutral:" + str(result[1]) + "Negative:" + str(result[2]) + "topic is: " + str(topic)
                         st.write(output)
            
                    else:
                     st.warning("Not sure! Try to add some more words")
    
    


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