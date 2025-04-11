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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

api_key = 'AIzaSyCaXxmYK9PPy0a18ZYhzyqDgKDn7AHquZI'
channel_id = 'UCK7AyP3Gg4ZURJgUbakZ1ug'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def get_uploads_playlist_id(youtube, channel_id):
    request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )
    response = request.execute()

    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return uploads_playlist_id

def get_all_video_ids(youtube, playlist_id):
    video_ids = []
    next_page_token = None
    total_result = None

    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        if total_result is None:
            total_result = response['pageInfo']['totalResults']

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_ids, total_result

def get_video_stats(youtube, videos_id):
    stats = []
    batch_size = 50
    all_data = []

    for i in range(0, len(videos_id), batch_size):
        batch_ids = videos_id[i:i + batch_size]
        request = youtube.videos().list(
            part='snippet, contentDetails, statistics',
            id=','.join(batch_ids)
        )
        response = request.execute()
        stats.extend(response.get('items', []))  # Gunakan .get() untuk mencegah error jika 'items' tidak ada

    for i in range(len(stats)):
        video_id = stats[i]['id']
        data = dict(
            video_id=video_id,
            videoTitle=stats[i]['snippet']['title'],
            published=stats[i]['snippet']['publishedAt'],
            description=stats[i]['snippet']['description'],
            Duration=stats[i]['contentDetails']['duration'],
            Definition=stats[i]['contentDetails']['definition'],
            Viewers=stats[i]['statistics'].get('viewCount', 0),
            Likes=stats[i]['statistics'].get('likeCount', 0),
            Comments=stats[i]['statistics'].get('commentCount', 0),
        )

        all_data.append(data)

    return all_data

def save_dataframe_to_excel(df, filename, index=False):
    try:
        df.to_excel(filename, index=index)
        print(f"DataFrame saved to {filename}")
    except Exception as e:
        print(f"Error saving DataFrame to Excel: {e}")


def main():
    uploads_playlist_id = get_uploads_playlist_id(youtube, channel_id)
    all_video_ids, total_result = get_all_video_ids(youtube, uploads_playlist_id)
    videos_id = all_video_ids
    videoStatistics = pd.DataFrame(get_video_stats(youtube, videos_id))

    df = pd.read_csv("cleaned_comments.csv")
    model = load_model("sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_len = 50
    sentiment_mapping = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    # Apply the mapping to create a new column
    df["classification"] = df["sentiment"].map(sentiment_mapping)
    # Sentiment summary
    df['weight'] = df['like_count'] + 1

    # Create pivot table to sum weights by video_id and classification
    summary = df.pivot_table(index='video_id',
                            columns='classification',
                            values='weight',
                            aggfunc='sum',
                            fill_value=0)

    # Optional: re-order columns
    summary = summary[['Positive', 'Neutral', 'Negative']]  # if all 3 exist
    summary = summary.reset_index()

    dataset = pd.merge(videoStatistics, summary, on="video_id", how="inner")
    save_dataframe_to_excel(dataset, "Dataset.xlsx")
    save_dataframe_to_excel(df, "Sentiment.xlsx")

if __name__ == "__main__":
    main()
