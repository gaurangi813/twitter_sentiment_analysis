import streamlit as st
import pickle
import re
import time
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Configuration
TWITTER_URL = "https://x.com/{}"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Download stopwords
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model files
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

# Alternative Twitter scraping method
def scrape_tweets(username, max_tweets=5):
    try:
        response = requests.get(TWITTER_URL.format(username), headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            tweets = []
            for tweet in soup.find_all('article', limit=max_tweets):
                text_div = tweet.find('div', {'data-testid': 'tweetText'})
                if text_div:
                    tweets.append({'text': text_div.get_text()})
            return {'tweets': tweets} if tweets else None
        return None
    except Exception as e:
        st.warning(f"Scraping error: {str(e)}")
        return None

# Sentiment analysis function
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vec = vectorizer.transform([text])
    return "Positive" if model.predict(text_vec)[0] == 1 else "Negative"

# UI Components
def tweet_card(tweet_text, sentiment):
    color = "#4CAF50" if sentiment == "Positive" else "#F44336"
    return f"""
    <div style="background-color:{color};padding:12px;border-radius:8px;margin:10px 0;color:white;">
        <h5 style="margin:0 0 8px 0;">{sentiment} Sentiment</h5>
        <p style="margin:0;">{tweet_text}</p>
    </div>
    """

def main():
    st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
    st.title("🐦 Twitter Sentiment Analyzer")
    
    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    
    if not model or not vectorizer:
        return

    # App interface
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Analyze Text", "Analyze Twitter User"]
    )

    if analysis_mode == "Analyze Text":
        st.subheader("Text Analysis")
        user_text = st.text_area("Enter text to analyze:")
        if st.button("Analyze Sentiment") and user_text.strip():
            sentiment = predict_sentiment(user_text, model, vectorizer, stop_words)
            st.markdown(tweet_card(user_text, sentiment), unsafe_allow_html=True)

    else:
        st.subheader("Twitter User Analysis")
        username = st.text_input("Twitter username (without @):", "Cristiano")
        tweet_count = st.slider("Number of tweets", 1, 5, 3)
        
        if st.button("Analyze Tweets") and username.strip():
            with st.spinner(f"Fetching {tweet_count} tweets from @{username}..."):
                tweets_data = scrape_tweets(username, tweet_count)
                
                if tweets_data and tweets_data['tweets']:
                    for tweet in tweets_data['tweets']:
                        text = tweet['text']
                        sentiment = predict_sentiment(text, model, vectorizer, stop_words)
                        st.markdown(tweet_card(text, sentiment), unsafe_allow_html=True)
                else:
                    st.error("Failed to fetch tweets. Possible solutions:")
                    st.markdown("""
                    1. **Use the Official Twitter API** (requires developer account)
                    2. **Try again later** - Twitter may be blocking temporary
                    3. **Use a VPN** if you're being rate-limited
                    4. **Check if the account exists**: [Twitter.com/{}](https://twitter.com/{})
                    """.format(username, username))
                    st.info("For reliable access, consider applying for a Twitter API key at developer.twitter.com")

if __name__ == "__main__":
    main()