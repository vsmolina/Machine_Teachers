import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("news_api")
newsapi = NewsApiClient(api_key=api_key)

class Sentiment_analysis():

    def __init__(self, Key_word):
        self.word = Key_word

    def fetch_all_news():
        headlines = newsapi.get_everything(
            q=self.word,
            language="en",
            page_size=100,
            sort_by="relevancy"
            )
        self.headlines = headlines