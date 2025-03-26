from openai import OpenAI
import time
from pathlib import Path
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, VideoFileClip, CompositeAudioClip, vfx, ColorClip, concatenate_audioclips
import requests
import moviepy.editor as mp
import math
from PIL import Image
import numpy
import random
from pydub import AudioSegment
import re
import nltk
from nltk.corpus import cmudict
import os
import shutil
from datetime import datetime
from pytube import YouTube
from pytube import Search
import asyncio
import io
import json
import requests

from rembg import remove

from pydub import AudioSegment
from pydub.silence import split_on_silence

from google.cloud import speech

import subprocess

client = OpenAI(api_key='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')


def createPng(path):
    # Load your image (replace 'path/to/your/image.jpg' with the actual file path)
    input_path = path
    output_path = '/Users/tassiloar/Desktop/Nexamedia/output.mov'

    with open(input_path, 'rb') as input_file:
        input_data = input_file.read()

    # Remove the background
    output_data = remove(input_data)

    # Save the output image
    with open(output_path, 'wb') as output_file:
        output_file.write(output_data)


def splitAudio():
    # Load your audio file
    audio = AudioSegment.from_file("/Users/tassiloar/Desktop/Nexamedia/audiotest.mp3")

    # Split audio on silence
    # min_silence_len: minimum length of silence (in ms) to consider for a split
    # silence_thresh: silence threshold (in dB); lower values mean more silence will be detected
    chunks = split_on_silence(audio,
        min_silence_len=200,  # Adjust this value to fit the minimum length of silence
        #silence_thresh=audio.dBFS-14,  # Adjust this value based on your audio file's characteristics
        silence_thresh=audio.dBFS-16,
        keep_silence=100,
        seek_step= 1
        )  # Optional: keeps 500ms of silence at the start and end of each chunk

    # Export the split audio files
    for i, chunk in enumerate(chunks):
        chunk.export(f"/Users/tassiloar/Desktop/Nexamedia/Audiotest/chunk{i}.mp3", format="mp3")



def createAudioOpenAI ():


    speech_file_path = Path("/Users/tassiloar/Desktop/Nexamedia/audiotesthd.mp3")
    response = client.audio.speech.create(
      model="tts-1",
      voice="onyx",
      input="""In 1994, Jeff Bezos founded Amazon.com in a garage in Bellevue, Washington, with a vision to create the most customer-centric company in the world. x14. Initially, Amazon was an online bookstore, leveraging the untapped potential of the internet. Bezos' insight into the exponential growth of web use drove his decision to diversify Amazon's offerings, leading to an expansive catalog of products and services.

Bezos' relentless focus on customer satisfaction, innovation, and long-term thinking set Amazon apart. The introduction of Amazon Prime in 2005 revolutionized online shopping with its speedy delivery, creating a loyal customer base. Amazon Web Services (AWS), launched in 2006, became a powerhouse in cloud computing, further diversifying Amazon's revenue streams.

Under Bezos' leadership, Amazon ventured into electronic devices, streaming services, and artificial intelligence with products like Kindle, Amazon Echo, and Alexa. These innovations not only solidified Amazon's position in the market but also opened new revenue channels.

Bezos' strategy of reinvesting profits into new ventures and technology led to exponential growth. His stake in Amazon, despite selling shares over time for personal investments, including his space exploration company Blue Origin, has made him one of the wealthiest individuals in history.

Jeff Bezos' journey from a garage startup to a global empire exemplifies the power of visionary leadership, relentless innovation, and customer obsession. It's a testament to how thinking big and taking bold risks can redefine industries and create unparalleled wealth.""",
      speed = 1.2
    )
  
    response.stream_to_file(speech_file_path)
    

import base64

def transcribe_speech(api_key, audio_file_path, language_code='en-US'):
    url = "https://speech.googleapis.com/v1/speech:recognize?key={}".format(api_key)
    
    # Read the audio file
    with io.open(audio_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    
    audio_content_base64 = base64.b64encode(audio_content).decode("utf-8")
    
    # Construct the request payload
    data = {
        "config": {
            "encoding": "MP3",
            "sampleRateHertz": 24000,
            "languageCode": language_code,
            "enableWordTimeOffsets": True,
        },
        "audio": {
            "content": audio_content_base64
        }
    }
    
    wordlist =[]
    
    # Make the request
    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    
    # Parse and print the results
    if response.status_code == 200:
        response_json = response.json()
        results = response_json.get('results', [])
        if not results:
            print("No transcription results")
        for result in results:
            alternatives = result.get('alternatives', [])
            for alternative in alternatives:
                words = alternative.get('words', [])
                for word_info in words:
                    word = word_info.get('word')
                    start_time = word_info.get('startTime')
                    end_time = word_info.get('endTime')
                    wordlist.append([word,start_time[:-1],end_time[:-1]])
                    #print(f"Word: {word}, start time: {start_time}")
    else:
        print('Error:', response.status_code, response.text) 
    
    return wordlist
        
        
def textVisualnew(words):
    
    texts = [];
    
    i = 0
    
    while i < len(words):
    
        text_clip1 = TextClip(words[i][0], 
                                fontsize=65, 
                                color="white", 
                                font="arial")
        
        text_clip_border = TextClip(words[i][0], 
                                fontsize=65, 
                                color="white", 
                                font="arial",
                                stroke_width=20,
                                stroke_color="black")
        
        text_clip1 = text_clip1.set_position(lambda t: ('center', 'center'), relative=True).set_position((12, 12))
        
        text_clip = CompositeVideoClip([text_clip_border, text_clip1])
        text_clip = text_clip.set_position(lambda t: ('center', text_clip.w/2 + 820))
        text_clip = text_clip.set_start(float(words[i][1]))
        text_clip = text_clip.set_end(float(words[i][2][:-1]))
        texts.append(text_clip) 
        
    
    return texts
    


def search_google_images(query, num = 1):
    """
    Search for images using Google Custom Search JSON API.

    :param query: Search query
    :param api_key: Your API key for Google Custom Search JSON API
    :param num: Number of search results to return (max 10 per request)
    :return: List of image URLs
    """
    
    if query == "":
      return "E"
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': '8798c246b4f534e50',
        'key': 'AIzaSyBMWrqHeJyxJkK3-Wi7CXEgjsrCzzRhP4U',
        'searchType': 'image',
        'num': num
    }

    try:
      response = requests.get(search_url, params=params)
      response.raise_for_status()  # Raises an HTTPError if the response was an error
      result = response.json()

      image_urls = []
      
      if 'items' in result:
        for item in result['items']:
          image_urls.append(item['link'])

      if not image_urls:
        print("No images found for the query.")
        
        words = query.split()
        words.pop(len(words)-1)
        word = ""
        for x in words:
          word = word + x

        return search_google_images(word, num)
      
      return image_urls
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "E"


def generate_unique_id():
    return random.randint(10000000, 99999999)


def download_video(query, path, initial_length=240, max_attempts=10):


                    yt = YouTube("https://www.youtube.com/shorts/55IriqTIDPA")
                    print(f"Downloading: {yt.title}")
                    stream = yt.streams.get_highest_resolution()
                    unique_id = generate_unique_id() 
                    download_path = os.path.join(path, f"{unique_id}.mp4")
                    print(download_path)
                    stream.download(output_path=path, filename=f"{unique_id}.mp4")
 
def aspectRatio(clip, set_width = 1080, set_height = 1920):
  
  height = clip.size[1]
  width = clip.size[0]
  
  clip = clip.resize(newsize=(int(width*(set_height/height)),int((set_height/height)*height)))
  
  height = clip.size[1]
  width = clip.size[0]
  
  if width<set_width:
    clip = clip.resize(newsize=(int(width*(set_width/width)),int(height*(set_width/width))))
  
    height = clip.size[1]
    width = clip.size[0]
    clip = clip.crop(x1=0, y1=(height-set_height)/2, x2=width, y2=height-(height-set_height)/2)
    
  else:
    height = clip.size[1]
    width = clip.size[0]
    clip = clip.crop(x1=(width-set_width)/2, y1=0, x2=width-(width-set_width)/2, y2=height)
  
  return clip

def addbackgrund(start_time, end_time):
  
        directory = "/Users/tassiloar/Desktop/Nexamedia/Media Library/Background_Video/"

        # List all files in the directory
        files = os.listdir(directory)

        # Select a random image file from the filtered list
        random_image_file = random.choice(files)

        # Create the full path to the image file
        background_path = os.path.join(directory, random_image_file)

        # Create an ImageClip
        background_clip = VideoFileClip(background_path)
        background_clip = aspectRatio(background_clip)
        
        while background_clip.duration < float(end_time):
          background_clip = CompositeVideoClip([background_clip.set_start(0), background_clip.set_start(background_clip.duration)])
        
        background_clip = background_clip.subclip(float(start_time),float(end_time))
        
        return background_clip


def remove_background_from_frame(frame_path, output_path):
    command = f"backgroundremover -i {frame_path} -o {output_path}"
    subprocess.run(command, shell=True)
    if os.path.exists(output_path):
        return output_path
    else:
        raise FileNotFoundError(f"Processed file {output_path} not found")
      
"""
def process_video(video_path, output_dir, output_video_path):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the video file
    clip = VideoFileClip(video_path)
    processed_clips = []

    # Iterate over each frame
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype="uint8")):
        frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")

        # Save the frame as a PNG
        ImageClip(frame).save_frame(frame_path)

        # Remove the background from the frame
        processed_frame_path = remove_background_from_frame(frame_path)

        # Load the processed frame and set its duration
        processed_clip = ImageClip(processed_frame_path).set_duration(1/clip.fps)
        processed_clips.append(processed_clip)

        # Optionally, remove the original frame to save space
        os.remove(frame_path)

    # Concatenate all processed image clips
    final_clip = CompositeVideoClip(processed_clips, method="compose")
    final_clip.write_videofile(output_video_path, fps=clip.fps)

    # Clean up: remove the processed frames
    for processed_clip in processed_clips:
        os.remove(processed_clip.filename)

# Usage
video_path = "/Users/tassiloar/Downloads/IMG_8953.mp4"
output_dir = "/Users/tassiloar/Desktop/test"
output_video_path = "/Users/tassiloar/Desktop/modified_video.mp4"
process_video(video_path, output_dir, output_video_path)

process_video("/Users/tassiloar/Downloads/IMG_8953.mp4")
"""


def createPng(path, output):


              
        #command = f"backgroundremover -i {path} -o {output}.mp4"
        command = f"backgroundremover -i {path} -tv -o {output}"
        subprocess.run(command, shell=True)

     


"""
yt = YouTube(vid)
#print(f"Downloading: {yt.length}")

stream = yt.streams.get_highest_resolution()
unique_id = generate_unique_id() 
download_path = os.path.join("/Users/tassiloar/Desktop/", f"{unique_id}.mp4")
print(download_path)
stream.download(output_path="/Users/tassiloar/Desktop/", filename=f"{unique_id}.mp4")
"""


def get_video_length(url):
    # Construct the youtube-dl command
    command = ["youtube-dl", "--dump-json", url]

    # Run the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
        return None

    # Parse the JSON output
    video_info = json.loads(result.stdout)

    # Extract the duration and convert it to an integer
    duration_seconds = int(video_info.get('duration', 0))

    return duration_seconds



def download():
  id = generate_unique_id()
  command = ['yt-dlp', '-o', f'/Users/tassiloar/Desktop/test/{id}.%(ext)s', 'https://www.youtube.com/watch?v=y69PJvjq_PA']

  try:
      subprocess.run(command, check=True)
  except subprocess.CalledProcessError as e:
      print(f"The command '{e.cmd}' failed with return code {e.returncode}")
      # Optionally, raise the error to exit the script or handle it differently
      raise RuntimeError("yt-dlp command failed!") from e
  
  return f'/Users/tassiloar/Desktop/test/{id}.webm'



def has_transparency(img_path):
     try:
        with Image.open(img_path) as img:
            # Ensure the image is in a format that supports transparency
            if img.mode == 'RGBA':
                # Check the alpha channel directly in RGBA mode
                alpha = img.getchannel('A')
                return any(pixel == 0 for pixel in alpha.getdata())
            elif img.mode == 'P':
                # Handle transparency in images with a palette (mode 'P')
                if 'transparency' in img.info:
                    # Check if the transparency index in the palette indicates full transparency
                    transparency = img.info['transparency']
                    if isinstance(transparency, bytes):
                        # For 8-bit paletted images, transparency is a byte sequence
                        return 0 in transparency
                    elif isinstance(transparency, int):
                        # Single transparent index, need to check if it corresponds to a fully transparent entry
                        palette = img.getpalette()[transparency * 3:transparency * 3 + 3]
                        return palette[-1] == 0
                else:
                    # No transparency data in the palette
                    return False
            else:
                # No alpha channel or palette transparency, no full transparency
                return False
     except IOError:
        print("Error in loading the image.")
        return False
  
  
def create_image_with_green_background(image_path):
    # Open the original image
    original_image = Image.open(image_path)

    # Get the size of the original image
    width, height = original_image.size

    # Create a new image with a green background
    green_background = Image.new('RGB', (width, height), (0, 255, 0))

    # Paste the original image onto the green background
    green_background.paste(original_image, (0, 0), original_image)

    # Save the result
    green_background.save(image_path, "PNG")
      

print(("hello")[1:4])