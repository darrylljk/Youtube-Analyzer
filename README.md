# Youtube Analyzer
Analyze YouTube comments with natural language processing and machine learning

## Try it out
- Access the web app here: [Youtube Analyzer](https://yt-analyzer.streamlit.app/) 🚀
- To run locally:
  - clone the repository: `git clone https://github.com/darrylljk/Youtube-Analyzer.git`
  - install dependencies: `pip install -r requirements.txt`
  - launch streamlit app: `streamlit run app.py`

## Features
- retrieve youtube video comments data with youtube api
- preprocessing with nltk (lemmatize, stem, tokenize, stopwords, lowercase)
- sentiment analysis with NLTK and plotly
- emotion classification with distilbert-base-uncased-emotion
- toxicity detection with unitary/toxic-bert
- topic analysis with gensim LDA model
- word cloud

## Screenshots
![yt-analyzer-streamlit-app-ss](https://github.com/user-attachments/assets/3098e95e-319f-4d99-8fb3-bdde67e48885)

## References
- youtube API: https://developers.google.com/youtube/v3/determine_quota_cost
  
## Contact
Darryl Lee - [LinkedIn](https://www.linkedin.com/in/darryl-lee-jk/)
