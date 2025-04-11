import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
import emoji
from nltk.stem import PorterStemmer, WordNetLemmatizer
from googleapiclient.discovery import build
from nltk.corpus import stopwords
import nltk
from googleapiclient.discovery import build
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Setup
api_key = 'AIzaSyCaXxmYK9PPy0a18ZYhzyqDgKDn7AHquZI'
channel_id = 'UCK7AyP3Gg4ZURJgUbakZ1ug'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def get_uploads_playlist_id():
    response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    return response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

def get_all_video_ids(playlist_id):
    video_ids = []
    next_page_token = None
    while True:
        response = youtube.playlistItems().list(
            part="contentDetails", playlistId=playlist_id, maxResults=50,
            pageToken=next_page_token).execute()
        video_ids += [item['contentDetails']['videoId'] for item in response['items']]
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return video_ids

def fetch_comments(video_ids):
    comments = []
    for video_id in video_ids:
        try:
            next_page_token = None
            while True:
                response = youtube.commentThreads().list(
                    part="snippet", videoId=video_id, maxResults=100,
                    pageToken=next_page_token).execute()
                for item in response['items']:
                    c = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'video_id': video_id,
                        'author': c['authorDisplayName'],
                        'published_at': c['publishedAt'],
                        'like_count': c['likeCount'],
                        'text': c['textDisplay']
                    })
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
        except Exception as e:
            print(f"Error fetching video {video_id}: {e}")
    return pd.DataFrame(comments)

# === Preprocessing ===

chat_words = { "gak": "tidak", "ga": "tidak", "lu": "kamu", "gue": "aku", "wkwkwk": "lol", "yg": "yang", "tdk": "tidak" }

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return text
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = emoji.replace_emoji(text, replace=' ')
    text = " ".join([chat_words.get(word, word) for word in text.split()])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

def classify_sentiment(text, pos_words, neg_words):
    words = text.split()
    negations = ["tidak", "bukan", "nggak", "gak", "no", "kurang"]
    sentiment = 0
    for i, word in enumerate(words):
        if word in pos_words:
            sentiment = -1 if i > 0 and words[i - 1] in negations else 1
        elif word in neg_words:
            sentiment = 1 if i > 0 and words[i - 1] in negations else -1
    return sentiment

def main():
    playlist_id = get_uploads_playlist_id()
    video_ids = get_all_video_ids(playlist_id)
    df = fetch_comments(video_ids)

    df = df[df['author'] != '@Exoraz']
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned_text'].notnull() & (df['cleaned_text'] != '')]

    pos_words = [line.strip() for line in open("positive.txt", encoding="utf-8")]
    neg_words = [line.strip() for line in open("negative.txt", encoding="utf-8")]

    df["sentiment"] = df["cleaned_text"].apply(lambda x: classify_sentiment(x, pos_words, neg_words))

    df.to_csv("cleaned_comments.csv", index=False)
    print("Saved cleaned_comments.csv")
    #print(df)

if __name__ == "__main__":
    main()
