import os
from pathlib import Path
import pandas as pd
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
load_dotenv()
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
api_key = os.getenv("news_api")
newsapi = NewsApiClient(api_key=api_key)

class Vader_analysis():

    def __init__(self, key_word):
        self.word = key_word

    def fetch_all_news(self):
        headlines = newsapi.get_everything(
            q=self.word,
            language="en",
            page_size=100,
            sort_by="relevancy"
            )
        self.headlines = headlines

    def stock_sentiment_score_df(self):
        sentiments = []

        for article in self.headlines["articles"]:
            try:
                text = article["content"]
                date = article["publishedAt"][:10]
                sentiment = analyzer.polarity_scores(text)
                compound = sentiment["compound"]
                pos = sentiment["pos"]
                neu = sentiment["neu"]
                neg = sentiment["neg"]
                
                sentiments.append({
                    "text": text,
                    "date": date,
                    "compound": compound,
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu
                    
                })
                
            except AttributeError:
                pass
            
        # Create DataFrame
        df = pd.DataFrame(sentiments)

        # Reorder DataFrame columns
        cols = ["date", "text", "compound", "positive", "negative", "neutral"]
        df = df[cols]
        self.df = df

    def descriptive_stats_df(self):
        self.df.describe()
