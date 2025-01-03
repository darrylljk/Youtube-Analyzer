from googleapiclient.discovery import build
import pandas as pd

def fetch_comments(api_key, video_url):
    """
    fetches comments from a youtube video using the youtube data api.

    parameters:
        api_key (str): your youtube data api key.
        video_url (str): url of the youtube video.

    returns:
        dataframe: a dataframe containing comments and metadata.
    """
    # extract the video id from the url
    video_id = video_url.split("v=")[1].split("&")[0]
    youtube = build('youtube', 'v3', developerKey=api_key)

    comments = []
    # send api request to fetch comments
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100) # set max results limit
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
        # handle pagination to fetch all comments
        request = youtube.commentThreads().list_next(request, response)

    return pd.DataFrame(comments)
