from googleapiclient.discovery import build
import pandas as pd

def fetch_comments(api_key, video_url):
    """
    fetches comments from a YouTube video using the YouTube Data API.

    parameters:
        api_key (str): Your YouTube Data API key.
        video_url (str): URL of the YouTube video.

    returns:
        df: df containing comments and metadata.
    """
    video_id = video_url.split("v=")[1].split("&")[0]
    youtube = build('youtube', 'v3', developerKey=api_key)

    comments = []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100)
    while request:
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'likes': comment['likeCount'],
                'published': comment['publishedAt']
            })
        request = youtube.commentThreads().list_next(request, response)

    return pd.DataFrame(comments)