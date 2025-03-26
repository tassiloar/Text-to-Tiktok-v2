import sys
import random

from googleapiclient.discovery import build
from pytubefix import YouTube

import global_var

api_key = global_var.GOOGLE_API_KEY

# Create a YouTube service object
youtube = build('youtube', 'v3', developerKey=api_key)

# search_youtube
# Input: Query string that is the topic of the video, max results
# Function: Generate a list of possibel videos
# Output: A list of videos for that string
def search_youtube(query, max_results=5):
    try:
        # Search for videos using the YouTube API
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",  
            videoDuration="medium", #4 to 20m
            order="relevance",
            #videoLicense= "creativeCommon",
            maxResults=max_results
        )
        
        response = request.execute()

    except ConnectionError:
        print()
        print("""\nERROR: A network error occurred. Please check your 
              internet connection.""")
        sys.exit(1)
        
    except Exception as e:
        # Handle any other errors
        print()
        print(f"\nERROR: An unexpected error occurred: \n{e}")
        sys.exit(1)

        
    video_urls = []
    # Process and print the search results
    for item in response['items']:
        video_urls.append(f"https://www.youtube.com/watch?v={item['id']['videoId']}")
    
    # Mix the order
    random.shuffle(video_urls)
    
    return video_urls


# get_best_stream
# Input: A youtube object and target resolution in format '720p'
# Function: Get the best stream
# Output: The best video stream
def get_best_stream(yt, target_resolution):
    # Get all video streams, excluding audio-only streams
    video_streams = yt.streams.filter(file_extension='mp4', 
                            progressive = True).order_by('resolution').desc()

    # Try to find the stream that matches the desired resolution
    for stream in video_streams:
        if stream.resolution == target_resolution:
            return stream

    # If the target resolution is not available, return the next best (highest 
    # available) resolution
    return video_streams.first()  


# get_best_stream
# Input: A list or urls, the current scene num and
# a target resolution in format '720p'
# Function: Downlod a youtube video
# Output: Null, downloads video to ./TEMP_content
def download_youtube_video(urls, scene_num, resolution):
    for url in urls:
        try:
            # Create a YouTube object for the video
            yt = YouTube(url)
            
            # Get the highest resolution stream available
            stream = get_best_stream(yt, resolution)
            
            # Download the video
            print(f"Downloading: {yt.title}")
            stream.download(output_path="./TEMP_data/TEMP_content", 
                            filename=f"{scene_num}.mp4")
            print("Download completed!")
            
            # Stop after the first successful download
            break
        
        except Exception as e:
            print(f"\nAn error occurred while downloading {url}:\n{e}")
            print("Moving to the next video...")