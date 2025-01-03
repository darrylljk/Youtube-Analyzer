import streamlit as st
from scraper import fetch_comments
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from transformers import pipeline
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('stopwords') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

# set up the page
st.set_page_config(page_title="YouTube Analyzer", layout="wide")

# color mapping for sentiment analysis
SENTIMENT_COLOR_MAP = {
    'very positive': '#00CC96',  # green
    'positive': '#636EFA',       # blue
    'neutral': '#AB63FA',        # purple
    'negative': '#FFA15A',       # orange
    'very negative': '#EF553B'   # red
}

# color mapping for emotion
set2_colors = px.colors.qualitative.Set2
emotion_color_map = {
    'joy': set2_colors[0],      
    'anger': set2_colors[1],    
    'sadness': set2_colors[2],   
    'fear': set2_colors[3],      
    'surprise': set2_colors[4],  
    'love': set2_colors[5]      
}

# -------------------------------------------------------
# functions
# -------------------------------------------------------

# cache the fetch_comments function
@st.cache_data
def cached_fetch_comments(api_key, video_url):
    return fetch_comments(api_key, video_url)

# cache analyze_sentiments
@st.cache_data
def cached_analyze_sentiments(comments):
    return analyze_sentiments(comments)

# load toxicity classifier from huggingface - unitary/toxic-bert
@st.cache_data
def load_toxicity_classifier():
    """
    loads the toxicity detection pipeline from hf.
    cached to avoid reloading the model repeatedly.
    """
    from transformers import pipeline
    return pipeline("text-classification", model="unitary/toxic-bert")

# load emotion classifier from huggingface - bhadresh-savani/distilbert-base-uncased-emotion
@st.cache_resource
def load_emotion_classifier():
    """
    loads the emotion classification pipeline from hf.
    cached to avoid reloading the model repeatedly.
    """
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# data preprocessing
def preprocess_comments(df):
    """
    preprocesses the text column in a DataFrame by tokenizing, converting to lowercase, removing stopwords, and filtering non-alphanumeric tokens.

    parameters:
        df (DataFrame): The DataFrame containing the text to preprocess.

    returns:
        dataFrame: The input DataFrame with an additional 'preprocessed_text' column.
    """
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('stopwords') 
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # define stopwords
    stop_words = set(stopwords.words('english')) | {"39", "quot", "br", "game"}
    
    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # apply preprocessing
    df['preprocessed_text'] = df['text'].apply(
        lambda text: ' '.join(
            lemmatizer.lemmatize(word) for word in word_tokenize(text.lower())
            if word.isalnum() and word not in stop_words
        )
    )
    return df

# analyze sentiments - generate sentiment score
def analyze_sentiments(comments):
    """
    analyzes the sentiment of each comment and assigns a sentiment level and score.

    parameters:
        comments (list of str): a list of comment strings.

    returns:
        list of dict: each dictionary contains 'sentiment' (str) and 'score' (int) for the comment.
    """
    sia = SentimentIntensityAnalyzer()
    results = []
    for comment in comments:
        # get sentiment score
        score = sia.polarity_scores(comment)['compound']
        if score > 0.6:
            sentiment = 'very positive'
            sentiment_score = 5
        elif score > 0:
            sentiment = 'positive'
            sentiment_score = 4
        elif score == 0:
            sentiment = 'neutral'
            sentiment_score = 3
        elif score > -0.6:
            sentiment = 'negative'
            sentiment_score = 2
        else:
            sentiment = 'very negative'
            sentiment_score = 1
        
        # append sentiment and score
        results.append({'sentiment': sentiment, 'score': sentiment_score})
    return results

# plot sentiments distribution with plotly
def plot_sentiments_plotly(df):
    """
    plots the distribution of sentiments in the dataframe using plotly, formatted for streamlit.

    parameters:
        df (DataFrame): a dataframe containing 'sentiment', 'sentiment_score', and 'text' columns.
    """
    # map sentiments to their desired order
    sentiment_order = ['very positive', 'positive', 'neutral', 'negative', 'very negative']
    score_mapping = {sentiment: score for score, sentiment in enumerate(sentiment_order[::-1], start=1)}

    # custom color mapping for sentiments
    sentiment_colors = SENTIMENT_COLOR_MAP

    # sort the dataframe by sentiment score
    df['sentiment'] = pd.Categorical(df['sentiment'], categories=sentiment_order, ordered=True)
    sentiment_counts = df.groupby(['sentiment'], as_index=False, observed=True).size()
    sentiment_counts['score'] = sentiment_counts['sentiment'].map(score_mapping)

    # sort the chart by sentiment score
    sentiment_counts = sentiment_counts.sort_values(by='score', ascending=True)

    # create the plotly bar chart with custom colors
    fig = px.bar(
        sentiment_counts,
        x='sentiment',
        y='size',
        color='sentiment',
        # title='Sentiment Distribution',
        labels={'sentiment': 'Sentiment', 'size': 'Number of Comments'},
        text='size',
        color_discrete_map=sentiment_colors  # apply custom color mapping
    )

    fig.update_traces(textposition='outside', marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='No of comments',
        template='plotly_white'
    )

    # display the chart in streamlit
    st.plotly_chart(fig, use_container_width=True)

# plot sentiment trend over time
def plot_sentiment_trend(df):
    """
    plots sentiment trends over time using timestamps in the dataframe.

    parameters:
        df (DataFrame): A dataframe containing 'published' and 'sentiment' columns.
    """
    # Ensure 'published' exists and is in datetime format
    if 'published' not in df.columns:
        st.error("Error: The 'published' column is missing from the data. Ensure timestamps are included in the fetch_comments function.")
        return

    df['published'] = pd.to_datetime(df['published'], errors='coerce')

    # Drop rows with invalid datetime values
    df = df.dropna(subset=['published'])

    # Group by time intervals and sentiment
    trend_data = df.groupby([pd.Grouper(key='published', freq='d'), 'sentiment']).size().reset_index(name='count')

    # Create the line chart
    fig = px.line(
        trend_data,
        x='published',
        y='count',
        color='sentiment',
        # title='Sentiment Trend Over Time',
        labels={'published': 'Time', 'count': 'Number of Comments', 'sentiment': 'Sentiment'},
        color_discrete_map=SENTIMENT_COLOR_MAP,
        # line_shape='spline'
    )
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='No of comments',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# generate word clouds
def generate_word_cloud(df, sentiment=None):
    """
    generates a word cloud from the comments in the dataframe.

    parameters:
        df (DataFrame): a dataframe containing 'text' and 'sentiment' columns.
        sentiment (str, optional): filter comments by sentiment (e.g., 'positive', 'negative'). if none, use all comments.
    """
    # filter comments by sentiment if specified
    if sentiment:
        filtered_df = df[df['sentiment'] == sentiment]
        text = ' '.join(filtered_df['preprocessed_text'])
    else:
        text = ' '.join(df['preprocessed_text'])

    # generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # display the word cloud
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# display most liked comments
def display_top_comments(df):
    """
    displays the top liked and disliked comments from the dataframe.

    parameters:
        df (DataFrame): A dataframe containing 'text' and 'likes' columns.
    """
    # sort by likes
    most_liked = df.sort_values(by='likes', ascending=False).head(5)
    st.write("##### Most liked comments")
    st.write("Comments that received the highest number of likes")
    st.table(most_liked[['author', 'text', 'likes', 'sentiment', 'sentiment_score', 'emotion', 'emotion_scores']])

# content summarizer
@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_comments(comments, max_length=50, min_length=10):
    summarizer = load_summarizer()
    combined_text = " ".join(comments)  # Combine comments into a single string
    summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# -------------------------------------------------------
# streamlit app
# -------------------------------------------------------
# youtube api key 
API_KEY = api_key # streamlit
# API_KEY = "insert API key here" # local

# app title
st.title("YouTube Comments Analyzer")
st.write("This app helps you to analyze the comments section of any YouTube video.")
st.write("_To get started, paste a YouTube video URL in the sidebar, click Analyze, and wait (~15s) for the page to load._")

# input fields
st.sidebar.header("Input YouTube Video")
video_url = st.sidebar.text_input("Paste URL here",
                                  value="https://www.youtube.com/watch?v=v2d4nyTpwpw", # default video 
                                  help="Enter the URL of the YouTube video")

# run analysis
if st.sidebar.button("Analyze"):
    if not video_url:
        st.error("Please provide the video URL.")
    else:
        try:
            progress = st.progress(0)

            # embed the video
            st.write("##### YouTube video")
            # st.video(video_url)
            video_id = video_url.split("v=")[-1].split("&")[0]  # extract the video ID from the URL
            embed_code = f"""
            <div style="display: flex; justify-content: flex-start;">
                <iframe width="640" height="360" 
                        src="https://www.youtube.com/embed/{video_id}" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            """
            st.markdown(embed_code, unsafe_allow_html=True)
            st.write("")
            st.write("")
            progress.progress(10)

            # -------------------------------------------------------
            # fetch data
            # -------------------------------------------------------
            df = cached_fetch_comments(API_KEY, video_url)
            progress.progress(30)
            
            # -------------------------------------------------------
            # data preprocessing
            # -------------------------------------------------------          
            df = preprocess_comments(df)            
            progress.progress(50)

            # -------------------------------------------------------
            # add sentiment, emotion, toxicity score
            # -------------------------------------------------------
            sentiment_results = cached_analyze_sentiments(df['preprocessed_text'])

                # -------------------------------------------------------
                # sentiment score
                # -------------------------------------------------------
            df['sentiment'] = [result['sentiment'] for result in sentiment_results]
            df['sentiment_score'] = [result['score'] for result in sentiment_results]
            progress.progress(70)

                # -------------------------------------------------------
                # emotion score
                # -------------------------------------------------------
            emotion_classifier = load_emotion_classifier()
            emotions = []
            emotion_scores = []

            for text in df['preprocessed_text']:
                try:
                    result = emotion_classifier(text)
                    # get the dominant emotion
                    emotions.append(result[0]['label'])

                    # extract all emotion scores
                    emotion_scores.append(result[0]['score'])

                except Exception as e:
                    emotions.append("Unknown")
                    emotion_scores.append({})  

            df['emotion'] = emotions
            df['emotion_scores'] = emotion_scores
            progress.progress(90)

                # -------------------------------------------------------
                # toxicity score
                # -------------------------------------------------------
            toxicity_classifier = load_toxicity_classifier()
            toxic_labels = []
            toxic_scores = []

            for text in df['preprocessed_text']:
                try:
                    result = toxicity_classifier(text)
                    toxic_labels.append(result[0]['label'])  # e.g., 'toxic' or 'non-toxic'
                    toxic_scores.append(result[0]['score'])  
                except Exception as e:
                    toxic_labels.append("Unknown")
                    toxic_scores.append(None)  

            df['toxic'] = toxic_labels
            df['toxic_score'] = toxic_scores

                # -------------------------------------------------------
                # length of comment
                # -------------------------------------------------------            
            df['comment_length'] = df['text'].apply(len)

            # -------------------------------------------------------
            # create dataframe
            # -------------------------------------------------------
            st.markdown("""
            <style>
            .dataframe {
                width: 100% !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.write("##### Data Overview")
            st.write("YouTube comments have been processed using advanced text cleaning techniques, e.g. lemmatization and stop word removal. Each comment is analyzed and assigned sentiment, emotion, and toxicity scores for deeper insights.")
            st.write("_Models used for scoring: sentiment (NLTK), emotion (distilbert-base-uncased-emotion), toxicity (unitary/toxic-bert)_")
            
            st.dataframe(df, use_container_width=True, height=250)
            st.write('')
            
            progress.progress(100)
            progress.empty()

            # -------------------------------------------------------
            # sentiment analysis
            # -------------------------------------------------------
            # layout: comments table and sentiment distribution
            cols = st.columns([2, 3])  # left column for table, right column for chart

            with cols[0]:
                # -------------------------------------------------------
                # sentiment distribution
                # -------------------------------------------------------
                st.write("##### Sentiment analysis")
                st.write("A breakdown of sentiments expressed in the comments")
                plot_sentiments_plotly(df)

            with cols[1]:
                # -------------------------------------------------------
                # sentiment trend
                # -------------------------------------------------------
                st.write("##### Sentiment trend")
                st.write("Visualizing how sentiment evolves over time")
                plot_sentiment_trend(df)

            # -------------------------------------------------------
            # most liked comments
            # -------------------------------------------------------
            display_top_comments(df)
            st.write('')

            # -------------------------------------------------------
            # emotion analysis
            # -------------------------------------------------------
            cols = st.columns([2, 3])  # left column for table, right column for chart

                # -------------------------------------------------------
                # emotion distribution
                # -------------------------------------------------------
            with cols[0]:
                st.write("##### Emotion analysis")
                st.write("A breakdown of emotions expressed in the comments")
                emotion_counts = df['emotion'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']

                fig = px.bar(
                    emotion_counts,
                    x='Emotion',
                    y='Count',
                    color='Emotion',
                    # title='Emotion Distribution',
                    labels={'Emotion': 'Emotion', 'Count': 'Number of Comments'},
                    text='Count',  
                    color_discrete_map=emotion_color_map
                    # color_discrete_sequence=px.colors.qualitative.Set2  
                )

                fig.update_traces(textposition='outside', marker=dict(line=dict(color='black', width=1)))
                fig.update_layout(
                    xaxis_title='Emotion',
                    yaxis_title='No of comments',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # -------------------------------------------------------
                # top comments by emotion
                # -------------------------------------------------------
            with cols[1]:
                st.write("##### Most emotional comments")
                st.write("Highlighting the top comment for each emotion category, such as joy, anger, and love, based on the highest emotion score")
                
                unique_emotions = df['emotion'].unique()
                most_emotion_comments = df.loc[df.groupby('emotion')['emotion_scores'].idxmax()]
                most_emotion_comments.reset_index()
                
                st.write(most_emotion_comments[['emotion', 'author', 'text', 'emotion_scores']])
            
            # -------------------------------------------------------
            # emotions vs likes boxplot
            # -------------------------------------------------------
            st.write("##### Popularity analysis: emotions vs likes")
            st.write("Exploring the relationship between the emotions expressed in comments and their popularity, measured by likes")
            fig = px.box(
                df,
                x='emotion',
                y='likes',
                color='emotion',
                color_discrete_map=emotion_color_map,
                # title="Emotion vs Likes",
                labels={'emotion': 'Emotion', 'likes': 'Number of Likes'},
                template='plotly_white'
            )
            fig.update_layout(xaxis_title='Emotion', yaxis_title='No of Likes')
            st.plotly_chart(fig, use_container_width=True)

            # -------------------------------------------------------
            # top comments by toxicity
            # -------------------------------------------------------
            st.write("##### Most toxic comments")
            st.write("Displaying the most toxic comments based on the toxicity score")

            if 'toxic_score' in df.columns:
                top_toxic_comments = df.sort_values(by='toxic_score', ascending=False).head(5)
                st.table(top_toxic_comments[['author', 'text', 'toxic_score']])
            else:
                st.warning("No toxicity data available.")
            st.write("")

            # -------------------------------------------------------
            # word cloud
            # -------------------------------------------------------
            st.write("##### Word clouds")
            st.write("Visual representations of the most frequently used words in comments, categorized by sentiment (size of each word reflects its frequency within the sentiment group) ")
            wordcloud_cols = st.columns(4)  # 4 equal columns for word clouds
            sentiments = ["very positive", "positive", "negative", "very negative"]
            for i, sentiment in enumerate(sentiments):
                with wordcloud_cols[i]:
                    st.write(f"{sentiment.capitalize()}")
                    generate_word_cloud(df, sentiment=sentiment)
            st.write("")

            # -------------------------------------------------------
            # topic modeling
            # -------------------------------------------------------
            st.write("##### Topic Modeling")
            st.write("Identifying key topics discussed in the comments using the Gensim LDA model. This analysis groups comments into distinct themes based on the words they contain.")
            num_topics = 5
            num_words = 5

            try:
                tokenized_text = df['preprocessed_text'].apply(str.split).tolist()
                dictionary = corpora.Dictionary(tokenized_text)
                corpus = [dictionary.doc2bow(text) for text in tokenized_text]
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10, alpha='symmetric', eta='auto')

                st.write("Generated topics:")
                for topic_num, topic_keywords in lda_model.print_topics(num_words=num_words):
                    # Extract only the keywords
                    keywords = [kw.split("*")[1].strip('"') for kw in topic_keywords.split(" + ")]
                    st.write(f"**Topic {topic_num + 1}**: {', '.join(keywords)}")

            except Exception as e:
                st.error(f"Error in topic modeling: {e}")
            
            st.write("")




        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------------------------------------
# Contact
# -------------------------------------------------------------
st.write('')
st.write('')
st.markdown("""
    <style>
        .footer {
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 12px;
            color: gray;
            margin-top: 20px;
        }
    </style>
    <div class="footer">
        Author: Darryl Lee | 
        <a href="https://www.linkedin.com/in/darryl-lee-jk/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/darrylljk" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)