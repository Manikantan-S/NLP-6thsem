import streamlit as st
import time
from movie_review_model import MovieReviewAnalyzer
import os

# Page config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .sentiment-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive {
        background-color: #90EE90;
        color: #006400;
    }
    .negative {
        background-color: #FFB6C1;
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MovieReviewAnalyzer()

# Check if model exists
model_exists = os.path.exists(os.path.join("saved_model", "sentiment_model.h5"))

# Title
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Model Status")
    if not model_exists:
        st.warning("Model not found. Please train the model.")
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes..."):
                accuracy = st.session_state.analyzer.train_model()
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
    else:
        st.success("Model is ready to use!")

# Main content
tab1, tab2 = st.tabs(["Single Review Analysis", "Rotten Tomatoes Analysis"])

with tab1:
    st.header("Analyze a Single Review")
    review_text = st.text_area("Enter your movie review:", height=150)
    
    if st.button("Analyze Review") and review_text:
        if not model_exists and st.session_state.analyzer.model is None:
            st.error("Please train the model first!")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence = st.session_state.analyzer.predict_review(review_text)
                
                # Display result
                st.markdown(f"""
                    <div class="sentiment-box {'positive' if sentiment == 'Positive' else 'negative'}">
                        <h3>Sentiment: {sentiment}</h3>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("Analyze Rotten Tomatoes Reviews")
    rt_url = st.text_input("Enter Rotten Tomatoes review URL:")
    
    if st.button("Fetch and Analyze Reviews") and rt_url:
        if not model_exists and st.session_state.analyzer.model is None:
            st.error("Please train the model first!")
        else:
            with st.spinner("Fetching and analyzing reviews..."):
                reviews = st.session_state.analyzer.get_rotten_tomatoes_reviews(rt_url)
                
                if reviews:
                    st.subheader("Retrieved Reviews")
                    for i, review in enumerate(reviews, 1):
                        st.text_area(f"Review #{i}", review, height=100, disabled=True)
                    
                    overall_sentiment, confidence = st.session_state.analyzer.analyze_reviews(reviews)
                    
                    if overall_sentiment:
                        st.subheader("Overall Sentiment Analysis")
                        st.markdown(f"""
                            <div class="sentiment-box {'positive' if overall_sentiment == 'Positive' else 'negative'}">
                                <h3>Overall Sentiment: {overall_sentiment}</h3>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Could not fetch reviews from the provided URL. Please check the URL and try again.")