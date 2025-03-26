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
import base64
import json
import io
import sys
from pytube.exceptions import VideoUnavailable
from rembg import remove
import textwrap
import subprocess
import spacy
import numpy as np
import cv2

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")
nltk.download('cmudict')
d = cmudict.dict()


# FUNCTIONS #########################################################################################

client = OpenAI(api_key='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')

used_links = []
#Input: Topic to generate text on
#Connects to openAI assistant
#Output: Text
def OpenAIAssistant(assistant_id, query):
  
  my_assistant = client.beta.assistants.retrieve(assistant_id)
  thread = client.beta.threads.create()

  message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = query
  )

  run = client.beta.threads.runs.create (
    thread_id = thread.id,
    assistant_id = my_assistant.id
  )

  while run.status != "completed":
   run = client.beta.threads.runs.retrieve (
     thread_id = thread.id,
     run_id = run.id
   )
   time.sleep(3)

  messages = client.beta.threads.messages.list (
    thread_id = thread.id
  )

  for message in messages.data:
    return message.content[0].text.value
    break


# Function to check if the input word is a noun
def is_noun_spacy(word):
    doc = nlp(word)
    # Check if the first (and likely only) word's POS tag is a noun
    return doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN"


  
#Input: Text, output_path
#Generates audio file sentnece by sentence using OpenAI TTS
#Output: null
def createAudioOpenAIold (sentences, output_path, OpenAIvoice):
  
  temp = []
    
  audios = []
  time = 0
  
  for i in range(len(sentences)): 
    tt = sentences[i]
    id = generate_unique_id()
    
    speech_file_path = Path(f"{output_path}{id}.mp3")
    response = client.audio.speech.create(
      model="tts-1",
      voice=OpenAIvoice,
      input=tt,
      speed = 1.2
    )
  
    response.stream_to_file(speech_file_path)
    
    audios.append(AudioFileClip(f"{output_path}{id}.mp3").set_start(time))
    temp.append((f"{output_path}{id}.mp3"))
    time += audios[i].duration

  print(temp)
  return audios
    

def createAudioOpenAI (text, output_path, OpenAIvoice):
  
  
  id = generate_unique_id()
    
  speech_file_path = Path(f"{output_path}{id}.mp3")
  response = client.audio.speech.create(
    model="tts-1-hd",
    voice=OpenAIvoice,
    input=text,
    speed = 1.3
  )
  
  response.stream_to_file(speech_file_path)
    
  audio = f"{output_path}{id}.mp3"

  return audio
    
    
#Input: Sentences, Audio files path, text colors
#Creates fast pace text basen on audio path
#Output: text clips 
def fastTextVisual(sentences, audios, secondary_color, primary_color, intro_sentence, font):

  audio_length = 0
  
  for audio in audios:
    audio_length+=audio.duration

  texts = []
  running_timer = 0
  
  color = primary_color
  x = 0
  re = 0
  
  while x < (len(sentences)):
    sylls = 0

    subtext = sentences[x]
    wordss = subtext.split()
    
    buffer = 10
    
    sylls += buffer*sum(word.endswith('.') for word in wordss)
    sylls += buffer*sum(word.endswith('!') for word in wordss)
    sylls += buffer*sum(word.endswith(':') for word in wordss)
    sylls += buffer*sum(word.endswith(';') for word in wordss)
    sylls += buffer*sum(word.endswith(',') for word in wordss)
    sylls += buffer*sum(word.endswith('?') for word in wordss)
    
    
    for word in wordss:
      word = word.replace(".", "")
      word = word.replace(",", "")
      word = word.replace("-", "")
      word = word.replace(":", "")
      word = word.replace("?", "")
      word = word.replace("!", "")
      word = word.replace(";", "")

      sylls += count_syllables(word)
    
   
    words = subtext.split()
    
    if " "+subtext == intro_sentence:
      color = secondary_color
      re = 2
     
    i = 0
    sum1 = 0
    
    while i < len(words): 
      qq=""
      second = ""
      second_time = 0
      dot = 0
      skip = False
      
      
      if words[i].endswith('.'):
        dot += buffer
        
      elif words[i].endswith(','):
        dot += buffer
        
      elif words[i].endswith(':'):
        dot += buffer
        
      elif words[i].endswith('?'):
        dot += buffer
        qq = "?"
        
      elif words[i].endswith('!'):
        dot += buffer
      
      elif words[i].endswith(';'):
        dot += buffer
      
      elif i != len(words)-1 and (len(words[i])+len(words[i+1])) < 13 and random.randint(1, 5) > 1:
        if ' - ' in words[i+1]:
          dot += buffer
        
        elif words[i+1].endswith('.'):
          dot += buffer
          
        elif words[i+1].endswith(','):
          dot += buffer
          
        elif words[i+1].endswith(':'):
          dot += buffer
          
        elif words[i+1].endswith('?'):
          dot += buffer
          qq = "?"
          
        elif words[i+1].endswith('!'):
          dot += buffer
        
        elif words[i+1].endswith(';'):
          dot += buffer
        
        words[i+1] = words[i+1].replace(".", "")
        words[i+1] = words[i+1].replace(",", "")
        words[i+1] = words[i+1].replace(":", "")
        words[i+1] = words[i+1].replace("?", "")
        words[i+1] = words[i+1].replace("-", "")
        words[i+1] = words[i+1].replace("!", "")
        words[i+1] = words[i+1].replace(";", "")
        
        second = words[i+1]
        second_time = count_syllables(words[i+1])
        skip = True
      
      if len(words[i]) > 9 and second == "" and re !=2:
        color = secondary_color
        re = 1
      
      words[i] = words[i].replace(".", "")
      words[i] = words[i].replace(",", "")
      words[i] = words[i].replace(":", "")
      words[i] = words[i].replace("?", "")
      words[i] = words[i].replace("!", "")
      words[i] = words[i].replace(";", "")
      words[i] = words[i].replace("-", "")
      
      #/Users/tassiloar/Downloads/the_bold_font/THE BOLD FONT - FREE VERSION - 2023.ttf
      text_clip1 = TextClip(words[i]+" "+second+qq, 
                            fontsize=65, 
                            color=color, 
                            font=font)
      
      text_clip_border = TextClip(words[i]+" "+second+qq, 
                            fontsize=65, 
                            color="black", 
                            font=font,
                            stroke_width=20,
                            stroke_color="black")
      
      text_clip1 = text_clip1.set_position(lambda t: (int((1080/2)-(text_clip1.size[0]/2)+12), int((1920/2)-(text_clip1.size[1]/2))+12+50))
      text_clip_border = text_clip_border.set_position(lambda t: (int((1080/2)-(text_clip_border.size[0]/2)), int((1920/2)-(text_clip_border.size[1]/2))+50))
      
      text_clip = CompositeVideoClip([text_clip_border, text_clip1], size=(1080, 1920))
      
      text_clip = text_clip.set_start(running_timer)
      text_clip = text_clip.set_end(running_timer+((count_syllables(words[i])+second_time+dot)* (audios[x].duration/sylls)))
      running_timer+= ((count_syllables(words[i])+second_time+dot)* (audios[x].duration/sylls))
      texts.append(text_clip) 
      
      sum1 +=count_syllables(words[i])+second_time+dot
      
      i+=1
      
      if skip == True:
        i+=1
      if re == 1 or i == len(words):
        color = primary_color
        re = 0

    
    x+=1

  k = 0
  while k < len(sentences):
      if sentences[k][-1] == '.':
          break
      k+=1
      
  first_duration = 0
  

  n = 0
  while n <= k:
    first_duration+= audios[n].duration
    n+=1

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


def download_image(image_url, path):
    """
    Download an image from a URL and save it to a local file.

    :param image_url: URL of the image to download
    :param path: Path where the image will be saved, including trailing slash
    """
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        
        # Check if the request was successful and the content type is an image
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            id = generate_unique_id()
            file_path = f"{path}{id}.png"

            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            # Validate the downloaded file
            if os.path.getsize(file_path) > 0:
                return file_path
            else:
                print(f"Downloaded file is empty or corrupted: {file_path}")
                os.remove(file_path)  # Remove the corrupted file
                return None
        else:
            print("Failed to download image: HTTP Status Code", response.status_code)
            return None
    except requests.RequestException as e:
        print("Request failed:", e)
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    

def search_and_download_image(query, path, num=3):
    

    while True:
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'q': query,
                'cx': '8798c246b4f534e50',
                'key': 'AIzaSyBMWrqHeJyxJkK3-Wi7CXEgjsrCzzRhP4U',
                'searchType': 'image',
                'num': num
                #'rights': '(cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived)',
            }

            response = requests.get(search_url, params=params)
            response.raise_for_status()
            result = response.json()

            image_urls = [item['link'] for item in result.get('items', []) if item['link'] not in used_links]

            for item in range(len(image_urls)):
                used_links.append(image_urls[item])  # Keep track of used links
                try:
                    response = requests.get(image_urls[item], stream=True, timeout=10)
                    if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                        id = generate_unique_id()
                        file_path = f"{path}{id}.png"
                        with open(file_path, 'wb') as file:
                            file.write(response.content)
                        if os.path.getsize(file_path) > 0:
                            print(image_urls[item])
                            return file_path
                except Exception as e:
                    continue
                
            if item == len(image_urls)-1:
              num*=2
              if num > 10:
                      if len(query.split()) > 2:  # Reduce the query complexity if possible
                        words = query.split()
                        words.pop(len(words)-2)
                        query = ' '.join(words)
                        num = 2

        except Exception as e:
            print(e)
            num *= 2
            if num > 10:
                if len(query.split()) > 2:  # Reduce the query complexity if possible
                    words = query.split()
                    words.pop(len(words)-2)
                    query = ' '.join(words)
                    num = 2  # Reset num
                else:
                    break  # Break out of the loop if the query cannot be simplified further
    print("Image Fail")
    print(query)
    xx = query.split()
    if len(xx)>1:
      xx.pop(len(xx)-2)
      print("Req img")
      strr = ' '.join(xx)
      num+=1
      return search_and_download_image(strr,path,num) # Return None if no image could be downloaded successfully
    else:
      print("Fail")
      return None


      
def download_video_old(query, path, min_time=240, next_vid_count=0):
    if query == "" or min_time >= 600 or next_vid_count >= 10:
        return "E"

    try:
        search = Search(query)
        if not search.results:
            time.sleep(2)
            print("Query failed, refining search terms.")
            refined_query = " ".join(query.split()[:-1])
            return download_video(refined_query, path, min_time, next_vid_count)

        for video in search.results:
            if video.length >= min_time:
                stream = video.streams.get_highest_resolution()
                unique_id = generate_unique_id()
                download_path = f"{path}/{unique_id}.mp4"
                stream.download(output_path=path, filename=f"{unique_id}.mp4")
                return download_path

        # If no suitable video found in the current result set, try the next set of results
        if next_vid_count < 15:
            return download_video(query, path, min_time + 60, 0)
        else:
            return download_video(query, path, min_time, next_vid_count + 1)

    except VideoUnavailable as e:
        print(f"An error occurred: {e}")
        if next_vid_count > 15:
            return download_video(query, path, min_time + 60, 0)
        else:
            return download_video(query, path, min_time, next_vid_count + 1)


#full pytube
def download_video_broken(query, path, initial_length=600, max_attempts=10):
    """
    Searches for and downloads a YouTube video based on a query and length criteria.
    
    :param query: The search query for finding videos.
    :param path: The directory path to save the downloaded video.
    :param initial_length: The initial maximum video length in seconds.
    :param max_attempts: Maximum number of attempts to modify the search query.
    """
    attempt = 0
    length = initial_length
    found_video = False

    while attempt < max_attempts and not found_video:
        time.sleep(2)
        print(f"Attempt {attempt+1}: Searching for videos with query '{query}' up to {length} seconds long.")
        s = Search(query)
        videos = [video for video in s.results if video.title not in used_links]
 
        for video in videos:
            try:
                yt = YouTube(video.watch_url)
                if yt.length < length and not "shorts" in video.watch_url:
                    
                    used_links.append(yt.title)
                    stream = yt.streams.get_highest_resolution()
                    unique_id = generate_unique_id() # Using YouTube video ID as a unique identifier
                    download_path = os.path.join(path, f"{unique_id}.mp4")
                    print(f"Downloading: {yt.title}")
                    stream.download(output_path=path, filename=f"{unique_id}.mp4")
                    print(f"Download Successful! Video saved to {download_path}")
                    found_video = True
                    return download_path
                     
            except Exception as e:
                print(f"Error downloading {video.watch_url}: {e}")
                continue

        # Adjust search criteria after checking all videos in the current batch
        if not found_video:
            length += 100  # Increase the acceptable video length
            attempt += 1
            if attempt % 3 == 0 and len(query.split()) > 1:  # Every third attempt, simplify the query if possible
                words = query.split()[:-1]
                query = ' '.join(words)

    if not found_video:
        print("Failed to download a suitable video after multiple attempts.")
        return None
  
  
#download using youtube-dl
def download_video(query, path, initial_length=300, max_attempts=10):
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
  
    """
    Searches for and downloads a YouTube video based on a query and length criteria.
    
    :param query: The search query for finding videos.
    :param path: The directory path to save the downloaded video.
    :param initial_length: The initial maximum video length in seconds.
    :param max_attempts: Maximum number of attempts to modify the search query.
    """
    attempt = 0
    length = initial_length
    found_video = False

    while attempt < max_attempts and not found_video:
        time.sleep(2)
        print(f"Attempt {attempt+1}: Searching for videos with query '{query}' up to {length} seconds long.")
        s = Search(query)
        videos = [video for video in s.results if video.title not in used_links]
 
        for video in videos:
            try:
                yt = (video.watch_url)
                used_links.append(yt)
                if get_video_length(yt) < length and not "shorts" in yt:
                    print(f"downloading {yt}")
                    unique_id = generate_unique_id() 
                    download_path = f'{path}{unique_id}.%(ext)s'
                    command = ['yt-dlp', '-o', f'{download_path}', f'{yt}']
                    try:
                      subprocess.run(command, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"The command '{e.cmd}' failed with return code {e.returncode}")
                        raise RuntimeError("yt-dlp command failed!") from e
                    
                    if not os.path.exists(f'{path}{unique_id}.webm'):
                      raise FileNotFoundError(f"The file {path}{unique_id}.webm does not exist.")
                    
                    print(f"Video downloaded successfully:{download_path}")
                    found_video = True
                    return f'{path}{unique_id}.webm'
                     
            except Exception as e:
                print(f"Error downloading {video.watch_url}: {e}")
                continue

        # Adjust search criteria after checking all videos in the current batch
        if not found_video:
            length += 100  # Increase the acceptable video length
            attempt += 1
            if attempt % 3 == 0 and len(query.split()) > 1:  # Every third attempt, simplify the query if possible
                words = query.split()[:-1]
                query = ' '.join(words)

    if not found_video:
        print("Failed to download a suitable video after multiple attempts.")
        return None

#Input: Text
#Splits text into punctuation sections
#Output: sentences array
def splitSentences(text):
    
  sentences = []
  current_sentence = ''
  sentence_end_chars = '.?!,;'  
  for char in text:
    current_sentence += char
    if char in sentence_end_chars:
      trimmed_sentence = current_sentence.strip(sentence_end_chars)
      if trimmed_sentence.strip():
        sentences.append(current_sentence.strip())
      current_sentence = ''

  if current_sentence.strip():
    sentences.append(current_sentence.strip())
  
  return sentences

#Input: word string
#Counts syllables in word
#Output: int syllables
def count_syllables(word):
    if word.lower() in d:
        return len([phoneme for phoneme in d[word.lower()][0] if phoneme[-1].isdigit()]) + 1
    else:
        return int(len(word)/3) + 1 + 1 


def generate_unique_id():
    return random.randint(10000000, 99999999)

a = 0
b = 0
c = 0

def createSceneSimple(instruction, start_time, end_time, words): 
    print(instruction)
    global a
    global b
    global c
    global ppp

    if a + b + c >=6:
      a = 0
      b = 0
      c = 0
  
    clips = [ColorClip(size=(10,10), color=(0, 0, 0), duration=1)]
  
    if instruction[0][0] == "V":
    
      instruction[0] = instruction[0][2:len(instruction[0])]
      if instruction[0][0] == ' ':
        instruction[0] = instruction[0][1:len(instruction[0])]
      
      inst = instruction[0] + " footage"
      link = download_video(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
      #print(link)
      clip = VideoFileClip(link)
      
      clip = clip.volumex(0.1)
      
      
      if float(clip.duration) - float(end_time) + float(start_time) -2 > 0:
        start = random.randint(0, int(float(clip.duration)) - int(float(end_time)) + int(float(start_time))-2)
        
      
      else:
        start = 0
      
      clip = clip.subclip(start, start+float(end_time)-float(start_time))
      

      
    else:  
      
      inst = instruction[0][2:len(instruction[0])]
      if inst == ' ':
        inst = inst[1:len(instruction[0])]
    
      link = search_and_download_image(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
      
      clip = ImageClip(link)
   
    
    if random.randint(1,3+a)<3:
        print("STYLE 1")
        clip = aspectRatio(clip, 1080, (random.randint(4,7)/10)*1920)
        clip = transitions(clip, (1080-clip.size[0])/2, (1920-clip.size[1])/2 + random.randint(-100,100), float(start_time))
        clips.append(clip)
        a+=2
     
    elif (clip.size[1] / clip.size[0]) > 1.5 and random.randint(1,2+b)<2:
      # and (is_image_split or (link.lower().endswith(('.mp4', '.mov')) and random.randint(1,2+b)<2))
      
        print("STYLE 2")
        clip = aspectRatio(clip, 1080, (random.randint(5,6)/10)*1920)
        
        split_position = clip.size[0] // 2  

        # Create the top half clip by cropping the original
        top_half = clip.crop(x1=0, x2=split_position)

        # Create the bottom half clip by cropping the original
        bottom_half = clip.crop(x1=split_position, x2=clip.size[0])
        
        buffer = 1920-(top_half.size[1]*2)
        
        if not buffer > 0:
          buffer = 0
        
        #top_half = top_half.set_position(('center', buffer//3))
        top_half = transitions(top_half, (1080-clip.size[0])/2, buffer//3, float(start_time))
        #bottom_half = bottom_half.set_position(('center',(buffer//3)+top_half.size[1]))
        bottom_half = transitions(bottom_half, (1080-clip.size[0])/2, (buffer//3)+top_half.size[1], float(start_time))
        
        clips.append(top_half)
        clips.append(bottom_half)
        b+=2
    else:
      print("STYLE 3")
      clip = aspectRatio(clip, 1080, 1920)
      clip = transitions(clip, (1080-clip.size[0])/2, (1920-clip.size[1])/2, float(start_time))
      clips.append(clip)
      c+=2
     


  
    #SIDE IMAGES
    for ff in range(len(instruction[1])):
      if instruction[1][ff][0] == ' ':
        instruction[1][ff] = instruction[1][ff][1:]
      inst = instruction[1][ff]+" PNG"
      link = search_and_download_image(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
      sizex = random.randint(700,800)
      x_pos = random.randint(-200,200)
      y_pos = random.randint(-50,200)
      rotate = random.randint(-20,20)
      starttt = 0.15 
      
      for wordx in words:
      
          if float(wordx[1]) >= float(start_time) and float(wordx[2]) <= float(end_time):
            wordar = wordx[0].split()
            for word in wordar:
              def stardaize(worddd):
                worddd = worddd.replace(',',' ')
                worddd = worddd.replace(". "," ")
                worddd = worddd.replace('! ',' ')
                worddd = worddd.replace('? ',' ')
                worddd = worddd.replace('"',' ')
                worddd = worddd.replace("%"," ")
                worddd = worddd.replace("#"," ")
                worddd = worddd.replace(',','')
                worddd = worddd.replace(". ","")
                worddd = worddd.replace('! ','')
                worddd = worddd.replace('? ','')
                worddd = worddd.replace('-',' ')
                worddd = worddd.replace(';',' ')
                worddd = worddd.lower()
                return worddd
          
              wordxx = stardaize(word)
              wordbase = stardaize(instruction[1][ff])
              print(wordxx)
              print(wordbase)
              
              if len(wordxx)>4 and len(wordbase)>4:
                
                if wordxx[1:4] == wordbase[1:4]:
                  starttt = -float(start_time) +float(wordx[1]) -0.2
                  print("      HEHEHEHE")
                  print(float(words[0][1]))
                  print(starttt)
                  
              else:
                if wordxx == wordbase:
                  starttt = -float(start_time) +float(wordx[1]) -0.2
                  print("      HEHEHEHE")
                  print(float(words[0][1]))
                  print(starttt)
                  
      clip = createimageclip(link, rotate, x_pos, y_pos,sizex,sizex, float(start_time)+starttt, float(end_time))  
          
      
      clip = clip.set_start(starttt)
      clips.append(clip)
  
   
    final_clip = CompositeVideoClip(clips, size=(1080, 1920))
    
    return final_clip

  
def createimageclip(link, rotate, x_pos, y_pos, wdith, height, start_time, end_time):
  
  def create_image_with_green_background(image_path):
    
    original_image = Image.open(image_path)

    # Ensure the image is in RGBA mode to handle transparency
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')

    # Get the size of the original image
    width, height = original_image.size

    # Create a new image with a green background
    green_background = Image.new('RGB', (width, height), (150, 150, 150))

    # Extract the alpha channel from the original image as a mask
    mask = original_image.split()[3]

    # Paste the original image onto the green background using the alpha channel as a mask
    green_background.paste(original_image, (0, 0), mask)

    # Save the result as PNG to maintain transparency
    green_background.save(image_path, "PNG")
  
  create_image_with_green_background(link)  
  link = createPng(link)
  print(link)
  print("ADDONS")
  
  clip = ImageClip(link)   
  #clip2 = ImageClip(link)                     
  
  if random.randint(1,2) == 1:
    clip = clip.rotate(lambda f: (rotate - f))     
  else:
    clip = clip.rotate(lambda f: (rotate + f))   
      
  clip = aspectRatio(clip, wdith, height)
  
  #clip2 = clip2.rotate(lambda f: (rotate + f)/2)
  
  #clip2 = aspectRatio(clip2, wdith+35, height+35)
  
  #clip2 = clip2.set_position((-17 ,-17))
  #clip2 = clip2.fl_image(lambda frame: 255 * np.ones(frame.shape, dtype=np.uint8))
  
  #clip = clip.set_start(float(start_time))
  #clip2 = clip2.set_start(float(start_time))
  
  #clip = CompositeVideoClip([clip2,clip])
  
  clip = transitions(clip, ((1080-clip.size[0])/2) + x_pos, ((1920-clip.size[0])/2) +y_pos, start_time, True)
 
  return clip
  

def aspectRatio(clip, set_width = 1080, set_height = 1920):
  
  height = clip.size[1]
  width = clip.size[0]
  
  clip = clip.resize(newsize=(int(width*(set_height/height)),int((set_height/height)*height)))
  
  height = clip.size[1]
  width = clip.size[0]
  
  if width<set_width:
    clip = clip.resize(newsize=(int(width*(set_width/width)),int(height*(set_width/width))))
  """
    height = clip.size[1]
    width = clip.size[0]
    clip = clip.crop(x1=0, y1=(height-set_height)/2, x2=width, y2=height-(height-set_height)/2)
    
  else:
    height = clip.size[1]
    width = clip.size[0]
    clip = clip.crop(x1=(width-set_width)/2, y1=0, x2=width-(width-set_width)/2, y2=height)
  """
  return clip


def textVisualnew(words):
    
    texts = [];
    
    i = 0
    
    while i < len(words):
    
        text_clip1 = TextClip(words[i][0], 
                                fontsize=65, 
                                color="white", 
                                font="Arial-bold")
        
        text_clip_border = TextClip(words[i][0], 
                                fontsize=65, 
                                color="white", 
                                font="Arial-bold",
                                stroke_width=20,
                                stroke_color="black")
        
        text_clip1 = text_clip1.set_position(lambda t: ('center', 'center'), relative=True).set_position((12, 12))
        
        text_clip = CompositeVideoClip([text_clip_border, text_clip1])
        text_clip = text_clip.set_position(lambda t: ('center', text_clip.w/2 + 820))
        text_clip = text_clip.set_start(float(words[i][1]))
        text_clip = text_clip.set_end(float(words[i][2]))
        texts.append(text_clip) 
        
        i+=1
        
    
    return texts


def transcribe_speech(audio_file_path, language_code='en-US'):
    url = "https://speech.googleapis.com/v1/speech:recognize?key={}".format("AIzaSyBMWrqHeJyxJkK3-Wi7CXEgjsrCzzRhP4U")
    
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

def match_speech(text, textgoogle):

  text = text.replace(',',' ')
  text = text.replace(". "," ")
  text = text.replace('! ',' ')
  text = text.replace('? ',' ')
  text = text.replace('"',' ')
  text = text.replace("%"," ")
  text = text.replace("#"," ")
  text = text.replace(',','')
  text = text.replace(". ","")
  text = text.replace('! ','')
  text = text.replace('? ','')
  text = text.replace('-',' ')
  text = text.replace(';',' ')


  words = text.split()
  
  print(words)
  print(textgoogle)


  h_to_l_textgoogle = sorted(textgoogle, key=lambda x: len(x[0]), reverse=True);
  
  indexs = []

  cur = 0

  #create ordered index list
  while cur < len(h_to_l_textgoogle):
      
      cur2 = 0

      while cur2 < len(textgoogle):
      
          if h_to_l_textgoogle[cur] == textgoogle[cur2]:
              indexs.append(cur2)
      
          cur2+=1
          
      cur+=1



  words = [[element, '', '',''] for element in words]
      

  n = 0
  
  while n <= 4:
      
      cur = 0

      while cur < int(len(indexs)):
          
          googleword = textgoogle[indexs[cur]][0].lower()
          
          upper = 99999999
          lower = 0
          
          cur2 = 0
          while cur2 < len(words):
              if words[cur2][3] != '':
                  
                  if words[cur2][3] <indexs[cur]:
                      lower = cur2
                  
                  if words[cur2][3] >indexs[cur]:
                      upper = cur2
                      break;
              
              cur2+=1
          
          cur2 = lower
          
          freq = 0
          freq_idx = 0
          
          while cur2 < len(words) and cur2 < upper:
              
              standardword = words[cur2][0].lower()

              if len(standardword)-1-n >= 4 and len(googleword)-1-n >= 4:
                  
                  standardword = standardword[0:8-n]         
                  googleword = googleword[0:8-n]

              if standardword == googleword:
                  freq+=1
                  
                  if freq>1:
                      words[freq_idx][1] = ''
                      words[freq_idx][2] = ''
                      words[freq_idx][3] = ''
                      break
                  else:
                      words[cur2][1] = textgoogle[indexs[cur]][1]
                      words[cur2][2] = textgoogle[indexs[cur]][2]
                      words[cur2][3] = indexs[cur]
                      freq_idx = cur2
              
              cur2+=1
          
          cur+=1
      n+=1

  return words


def merge_empties(words):
  x = 0

  
  while x < len(words):
    
    if words[x][1] == '':
          
      if x == 0:
        words[x+1][0] = words[x][0] +  ' ' + words[x+1][0]
        
      elif x == len(words)-1:
        words[x-1][0] =  words[x-1][0] +  ' ' + words[x][0] 
      
      elif words[x+1][0][0].isupper() == True:
        words[x-1][0] =  words[x-1][0] +  ' ' + words[x][0] 
        
      else:
        if len(words[x-1][0]) < len(words[x+1][0]):
          words[x-1][0] =  words[x-1][0] +  ' ' + words[x][0]
        else:
          words[x+1][0] = words[x][0] +  ' ' + words[x+1][0]
        
        if words[x+1][1] != '' and words[x-1][1] != '':
            words[x+1][1] = (float(words[x-1][2])+float(words[x+1][1]))/2
          
              
      words.pop(x)
      x-=1
    x+=1
  

  x = 0
  
  while x < len(words):

    if float(words[x][2]) - float(words[x][1]) <= 0.150 or len(words[x][0]) <= 4:
      
      if x != len(words)-1 and (x == 0 or words[x+1][0][0].isupper() != True) and len(words[x+1][0]) <= 8:
        
        words[x+1][0] = words[x][0] +  ' ' + words[x+1][0]
        words[x+1][1] = words[x][1]
        
        words.pop(x)
        x-=1
                
      elif len(words[x-1][0]) <= 8:

        words[x-1][0] =  words[x-1][0] +  ' ' + words[x][0]
        words[x-1][2] = words[x][2]
                  
        words.pop(x)
        x-=1
        
      elif float(words[x][1]) - float(words[x][2]) <= 0.11:
        
        if x == 0:   
          words[x][2] = float(words[x][2])+0.100
          words[x+1][1] = float(words[x+1][1])+0.100
        
        elif x == len(words)-1:
          words[x][1] = float(words[x][1])-0.100
          words[x-1][2] = float(words[x-1][2])-0.100
        else:
          words[x][1] = float(words[x][1])-0.100
          words[x-1][2] = float(words[x-1][2])-0.100
          
          words[x][2] = float(words[x][2])+0.100
          words[x+1][1] = float(words[x+1][1])+0.100
        
    x+=1


  x = 0
  while x < len(words)-1:

    if words[x][0][0].isupper() == True and x != 0:
      words[x][1]= float(words[x][1])+0.1
      words[x-1][2]= float(words[x-1][2])+0.1

    words[x][2] = words[x+1][1]

    x+=1
    
  return words

def find_section_bound(words, setions):
  
    section_times = []
    
    words_split = list(words)
    
    x = 0
    while x < len(words_split):
      wordsinside = words_split[x][0].split()
      
      x2 = 1
      for item in wordsinside:
        words_split.insert(x + x2, [item, words_split[x][1],words_split[x][2]])
        x2+=1
      
      words_split.pop(x)
      
      x+=1

    for section in setions:
      start = -1
      end = -1
      
      section = section.replace(',',' ')
      section = section.replace(". "," ")
      section = section.replace('! ',' ')
      section = section.replace('? ',' ')
      section = section.replace('"',' ')
      section = section.replace("%"," ")
      section = section.replace("#"," ")
      section = section.replace(',','')
      section = section.replace(". ","")
      section = section.replace('! ','')
      section = section.replace('? ','')
      section = section.replace('-',' ')
      section = section.replace(';',' ')
  
      setion_split = section.split()

      i = 0
      while i < len(words_split)-3:
          
        if words_split[i][0] == setion_split[0] and words_split[i+1][0] == setion_split[1] and words_split[i+2][0] == setion_split[2]:
          start = float(words_split[i][1])
        
        if words_split[i][0] == setion_split[len(setion_split)-3] and words_split[i+1][0] == setion_split[len(setion_split)-2] and words_split[i+2][0] == setion_split[len(setion_split)-1]:
          end = float(words_split[i+2][2])
        
        i+=1
      
      section_times.append([start,end])
    
    x = 0
    
    while x < len(section_times)-1:
      
      if int(section_times[x][1]) == -1 and int(section_times[x+1][0]) != -1 :
        section_times[x][1] = float(section_times[x+1][0])
      elif int(section_times[x+1][0]) == -1 and int(section_times[x][1]) != -1:
        section_times[x+1][0] = float(section_times[x][1])
      else:
        avg = (float(section_times[x][0]) + float(section_times[x+1][1]))/2
        section_times[x][1] = avg
        section_times[x+1][0] = avg
      
      x+=1
    
    section_times[0][0] = 0
    section_times[len(section_times)-1][1] = words_split[len(words_split)-1][2]
    
    return section_times
    

def instructionold(setions):
  
  order = [0]
  b = 0
  
  for section in sections:
    
    if order[len(order)-1] == 1 and b <= 8:
      b +=1
    if order[len(order)-1] == 2 and b <= 8:
      b -=1
    
    decider = random.randint(1,10)
    if decider > 5 + b:
      if order[len(order)-1] == 2:
        b = 0
      order.append(1)
    elif decider <= 5 + b:
      if order[len(order)-1] == 1:
        b = 0
      order.append(2)
  
  
  order.pop(0)
  
  ImgText = ""
  ImgCount = 0
  VidText = ""
  VidCount = 0
  
  x = 0
  while x < len(setions):
    if order[x] == 1:
      ImgText += setions[x]
      ImgCount+=1
    else:
      VidText += setions[x]
      VidCount+=1
    x+=1
    
  instructionsImg = f"{ImgText}  MUST RETURN {ImgCount} DIFFERENT VISUAL SETS AS THAT IS THE AMOUNT OF TEXT SECTIONS SEPERATED BY QUOTATION MARKS AND SEPERATE THESE VISUALS BY USING THE # CHARACTER"
  visualsImg = OpenAIAssistant("asst_UIcw4r18v6MTTfmAmVqOUv9S", instructionsImg)
  visualsImg = visualsImg.replace('"', '')
  visualsImg = visualsImg.replace("'", '')
  partsImg = visualsImg.split('#')
  
  instructionsVid = f"{VidText}  MUST RETURN {VidCount} DIFFERENT VISUAL SETS AS THAT IS THE AMOUNT OF TEXT SECTIONS SEPERATED BY QUOTATION MARKS AND SEPERATE THESE VISUALS BY USING THE # CHARACTER"
  instructionsVid = OpenAIAssistant("asst_A0OGS8FaHAUBLK1hA2DvUxA1", instructionsVid)
  instructionsVid = instructionsVid.replace('"', '')
  instructionsVid = instructionsVid.replace("'", '')
  partsVid = instructionsVid.split('#')
  
  instructions = []
  
  ImgCount = 0
  VidCount = 0
  
  x = 0
  while x < len(setions):
    if order[x] == 1:
      instructions.append("I "+partsImg[ImgCount])
      ImgCount+=1
    else:
      instructions.append("V "+partsVid[VidCount])
      VidCount+=1
    
    x+=1
  
  return instructions


def instructionold2(sections, title):
  
  instructions = []
  
  for section in sections:
    
    i = 0
    while i < len(section)-2:
        if section[i] == '?' or section[i] == '.' or section[i] == '!':
            section = section[:i+2] + section[i+2].lower() + section[i+2 + 1:]
        i+=1
    
    section = section.replace("?", "")
    section = section.replace(".", "")
    section = section.replace("!", "")
    section = section.replace(",", "")
    section = section.replace(";", "")
    section = section.replace("-", "")
    section = section.replace(":", "")
    section = section.replace('"', '')
    section = section.replace("'", '')
    
    
    primary = ""
    secondary = []
    
    words = section.split()
    
    ran = len(words) 
    i = 0
    while i < ran:
      
        if words[i][0] == '%':
          
            cur = words[i][1:]
            
            if i < len(words) - 1:
                if words[i+1][0].isupper()or words[i+1][0] == '%':
                    cur += " " +words[i+1]
                    del words[i+1]
                    ran-=1
                    
                primary = cur
                del words[i]
                ran-=1
                print("1")
                break
        i+=1
        
    del words[0]
    
    if primary == "":
      
      if i == 0:
        primary = title
      else:
        ran = len(words) 
        i = 0
        while i < ran:
        
          if words[i][0].isupper():
              
              cur = words[i]
          
              if i < len(words) - 1:
                  if words[i+1][0].isupper() or words[i+1][0] == '%':
                      cur += " " +words[i+1]
                      del words[i+1]
                      ran-=1
                      
                  primary = cur
                  del words[i]
                  ran-=1
          
          i+=1
        
    if primary != "":
        
      
      ran = len(words) 
      i = 0
      while i < ran:
        
        if words[i][0] == '%' and words[i][1:] not in secondary and words[i] != primary:
            
            cur = words[i][1:]

            if i < len(words) - 1:
                if words[i+1][0].isupper()or words[i+1][0] == '%':
                    cur += " " +words[i+1]
                    del words[i+1]
                    ran-=1
                    
                del words[i]
                ran-=1
                
                if " " in cur:
                    secondary.append(primary)
                    primary = cur
                else:
                    secondary.append(cur)
        i+=1
        
      ran = len(words) 
      i = 0
      while i < ran:
        
        if words[i][0].isupper() and words[i] not in secondary and words[i] != primary:
            
            cur = words[i]

            if i < len(words) - 1:
                if words[i+1][0].isupper()or words[i+1][0] == '%':
                    cur += " " +words[i+1]
                    del words[i+1]
                    ran-=1
                    
                del words[i]
                ran-=1
                
                secondary.append(cur)
        i+=1
  
    instructions.append([primary,secondary])
    
    
  #2nd part
  
  randomx = 0
  i = 0
  
  while i < len(instructions):
    x = random.randint(1,2)
    if x == randomx:
      x1 = random.randint(1,2)
      x2 = random.randint(1,2)
      if x1 != x:
        randomx = x1
    else:
      randomx = x
    
    if instructions[i][0] != "":
      if randomx == 1:
        instructions[i][0] = "I "+instructions[i][0]
      else:
        instructions[i][0] = "V "+instructions[i][0]
    else:
        l = "V "
        if x2 == 2:
          l = "I "
          
        if i == 0:
          instructions[i][0] = l+ title
        else:
         instructions[i][0] = l+ instructions[i-1][0][2:]
    
    i+=1
  
  return instructions

def instruction(sectionss, topic):
  
  sections = list(sectionss)
  
  instruction = []
  instructions = [0]
  
  b = 0
  
  for section in sections:
    
    if instructions[len(instructions)-1] == 1 and b <= 8:
      b +=1
    if instructions[len(instructions)-1] == 2 and b <= 8:
      b -=1
    
    decider = random.randint(1,10)
    if decider > 5 + b:
      if instructions[len(instructions)-1] == 2:
        b = 0
      instructions.append(["I "][0])
    elif decider <= 5 + b:
      if instructions[len(instructions)-1] == 1:
        b = 0
      instructions.append(["V "][0])
  
  instructions.pop(0)
  
  
  x = 0
  for section in sections:
    
    section= " "+section
    section = section.replace("?", "")
    section = section.replace(".", "")
    section = section.replace("!", "")
    section = section.replace(",", "")
    section = section.replace(";", "")
    section = section.replace(":", "")
    section = section.replace('"', '')
    section = section.replace("'", '')
    
    
    primary = []
    secondary = []
    
    primary.append(topic)
    
    words = section.split()
    
    ran = len(words) 
 
    i = 0
    
    while i < ran:
      
      if words[i][0] == '%':
        
        term = ""
        xx = 0
        ff = False
        while i+xx <ran and words[i+xx][len(words[i+xx])-1] != '%':
        
          term = term+" "+words[i]     
          
          if xx > 4:
            term = words[i]
            ff = True
            break
          xx+=1
        
        if not ff:
          term = term+" "+words[i+xx]
        
        term = term.replace('%','')   
        
        secondary.append(term)
        
        
        primary.append(term)
           
      i+=1
        
    cout = 0
        
    if instructions[x][0] == "I": 
      cout = 4
          
    elif instructions[x][0] == "V": 
      cout = 3
          
    i = 1
    lenx = len(primary)
    while lenx < cout and i < ran:
      found = False  # Flag to indicate if the substring is found
      for word in primary:
          if words[i][1:3].lower() in word.lower():  # Check if the substring is in the current word
              found = True
              break 
      if is_noun_spacy(words[i].lower()) == True and not found and words[i][0] != '%':
                
        if i < len(words) - 1:
          if words[i+1][0].isupper() and words[i+1][0] != '%':
             if i < len(words) - 2:
                if words[i+2][0].isupper() and words[i+2][0] != '%':
                  primary.append(words[i]+""+words[i+1]+""+words[i+2])
                else:
                  primary.append(words[i]+""+words[i+1])
             else:
              primary.append(words[i]+""+words[i+1])
            
          else:
            primary.append(words[i])
        else:
          primary.append(words[i])  
                    
        lenx+=1
        
      i+=1
        
    primary = ' '.join(primary)
    
    words = primary.split()
    unique_words = []
    seen_words = set()

    for word in words:
      if word.lower() not in seen_words:
          unique_words.append(word)
          seen_words.add(word.lower())

    primary = ' '.join(unique_words)
    primary = primary.replace('  ',' ')
    primary = primary.replace('  ',' ')
       
    instruction.append([instructions[x]+primary, secondary])
    
    x+=1     

  for item in instruction:
    if len(item[1]) == 0:
      if random.randint(1,2) == 1:
        item[1].append(topic)
  
  return instruction


def addbackgrund(typee, color, start_time, end_time):

        def adjust_rgb_background(clip, increase_by):
          def adjust_frame_colors(frame):
              # Ensure we don't exceed 255 after the increase
              return np.clip(frame + increase_by, 0, 255).astype(np.uint8)

          # Apply the color adjustment to each frame
          return clip.fl_image(adjust_frame_colors)

        color[0]-=100
        color[1]-=100
        color[2]-=100
        
        if typee == "gradient_motion":
          background_clip = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/bnlmotion.mp4")
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        elif typee == "lines_motion_black":
          background_clip = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/treeb.mp4")
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        elif typee == "lines_motion_white":
          background_clip = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/Sequence 01.mp4")
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        elif typee == "paper_motion":
          background_clip = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/paperback.mp4")
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        elif typee == "grainy_static":
          color[0]-=100
          color[1]-=100
          color[2]-=100
          background_clip = ImageClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/grainy.png")
          background_clip = background_clip.set_duration(float(end_time)-float(start_time))
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        else: #solid
          color[0]-=100
          color[1]-=100
          color[2]-=100
          background_clip = ImageClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/blank.png")
          background_clip = background_clip.set_duration(float(end_time)-float(start_time))
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          

        while background_clip.duration < float(end_time):
          background_clip = CompositeVideoClip([background_clip.set_start(0), background_clip.set_start(background_clip.duration)])
        
        background_clip = background_clip.subclip(float(start_time),float(end_time))
        background_clip = aspectRatio(background_clip)
        
        return background_clip
   

def createPng(path):

 
        command = f"backgroundremover -i {path} -o {path[:-4]}pp.png"
        subprocess.run(command, shell=True)
        return path[:-4]+"pp"+".png"

    
  
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

def is_image_split(image_path, threshold=10, scan_width=3):
  
    if image_path.lower().endswith(('.mp4', '.mov')):
      return False
    
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not loaded properly. Check the file path.")

    # Calculate the midpoint of the width of the image
    mid_x = image.shape[1] // 2

    # Define the area around the midpoint we want to scan
    scan_area = range(mid_x - scan_width, mid_x + scan_width + 1)

    # Initialize the count of matching pixel colors
    match_count = 0

    # Iterate over the height of the image
    for y in range(image.shape[0]):
        # Check the colors of pixels just to the left and right of the midpoint
        for x in scan_area:
            # Make sure we don't go out of image bounds
            if x <= 0 or x >= image.shape[1] - 1:
                continue

            # Get the pixel color values to the left and right of the midpoint
            pixel_left = image[y, x - 1]
            pixel_right = image[y, x + 1]

            # Check if the pixel colors are similar within a threshold
            if np.all(np.abs(pixel_left.astype(int) - pixel_right.astype(int)) < threshold):
                match_count += 1

    # Determine if the number of matching pixels is above a certain percentage of the image height
    # If it is, we assume there is no split
    if match_count > (image.shape[0] * 0.5 * len(scan_area)):
        return False
    else:
        return True
  

def OpenShot(topic, end_time):
  
  text = OpenAIAssistant("asst_cwEYqXNpyqSRMvScn3uTGt1M", topic)
  
  text = text.replace(' "', '')
  text = text.replace(" '", '')
  text = text.replace('" ', '')
  text = text.replace("' ", '')
  temp = text.split('#')
  
  title = temp[0]
  img = temp[1]
  
  title = textwrap.fill(title, width=10)
  
  random_num = 3
  
  if 1 == 1:
    
      img+=" PNG"
    
      
      link = search_and_download_image(img, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
      print(link)
      
      clip = ImageClip(link)
      
      if random_num == 3:
        kp = os.path.getsize(link)
        createPng(link)
        if kp/(os.path.getsize(link[:-4]+"pp"+".png")) > 3:
          random_num = random.randint(1,2)
        else:
          clip = ImageClip(link[:-4]+"pp"+".png")
      
      clip = clip.set_start(0)
      clip= clip.set_duration(end_time)
      clip = clip.set_end(float(end_time))
      
    
      if random_num == 1 :
        clip = aspectRatio(clip)
      elif random_num == 2 or random_num == 3:
        clip = aspectRatio(clip, 1080, (random.randint(3,7)/10)*1920)
  
      text_clip1 = TextClip(title, 
                                fontsize=140, 
                                color="white", 
                                font="Arial-bold")
        
      text_clip_border = TextClip(title, 
                                fontsize=140, 
                                color="white", 
                                font="Arial-bold",
                                stroke_width=20,
                                stroke_color="black")
        
      text_clip1 = text_clip1.set_position(lambda t: ('center', 'center'), relative=True).set_position((12, 12))
        
      text_clip = CompositeVideoClip([text_clip_border, text_clip1])
      
  
  text_clip.set_start(0)
  text_clip.set_duration(end_time)
  text_clip.set_end(end_time)
  
  clip = clip.set_position(lambda t: ('center', text_clip.size[1]))
  text_clip = text_clip.set_position(lambda t: ('center', 100))
  
  clip = CompositeVideoClip([text_clip,clip], size=(1080, 1920))
  
  clip.set_start(0)
  clip.set_end(end_time)
  clip= clip.set_duration(end_time)
  
  
  return clip
  

def transitions(clip, x_pos, y_pos, start, ispopup = False):
  
    duration_swipe = 0.15
    pos_dict = [
            lambda t: (min(0, 1080 * (t / duration_swipe - 1))+x_pos, y_pos),
            lambda t: (max(0, 1080 * (1 - t / duration_swipe)+x_pos), y_pos),
            lambda t: (x_pos, min(0, 1920 * (t / duration_swipe - 1))+y_pos),
            lambda t: (x_pos, max(0, 1920 * (1 - t / duration_swipe))+y_pos),
      ]
    
    x = random.randint(2,4)
    if x ==2 :
      clip = clip.set_position(pos_dict[random.randint(0,len(pos_dict)-1)])
      
      if random.randint(1,1) == 1:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/wooosh1.mp4")
      else:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/woosh2.mp4")
      
      sound_effects.append(video.audio.set_start(start).volumex(0.04))
      
      
      
    elif x == 3 and ispopup == False:
      if clip.mask is None:
            clip = clip.set_mask(ColorClip(clip.size, color=1, ismask=True, duration=clip.duration))

      clip.mask = clip.mask.fx(vfx.fadein, 0.75)
      
      clip = clip.set_position((x_pos,y_pos))
    
    elif x == 4:
      video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/pop.mp4")
      sound_effects.append(video.audio.set_start(start).volumex(0.02))
      
      clip = clip.set_position((x_pos,y_pos))

    else:
      clip = clip.set_position(pos_dict[random.randint(0,len(pos_dict)-1)])
      
      if random.randint(1,1) == 1:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/wooosh1.mp4")
      else:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/woosh2.mp4")
      
      sound_effects.append(video.audio.set_start(start).volumex(0.05))
      
    return clip
  

"""
topic = "Kylie Jenner"

text = OpenAIAssistant("asst_7zYo27j8yxn6xLBdcM5rzAJr", f"Follow your instructions to create a new text on {topic} between 210 and 230 words long")

print("text")
print(text)
print()

sections = re.findall(r'"(.*?)"',text)

print("sections")
print(sections)
print()

instructions = instruction(sections,topic)

print(instructions)

print(len(instructions))
print(len(sections))

text = text.replace('%','')

audio = createAudioOpenAI(text, "/Users/tassiloar/Desktop/Nexamedia/HEAP/", "onyx")
audio_clip = AudioFileClip(audio).set_start(0)

print("audio")
print(audio)
print()

spt = transcribe_speech(audio, language_code='en-US')
spt = match_speech(text, spt)
spt = merge_empties(spt)

print("spt")
print(spt)
print()

"""


topic = "Kylie Jenner"

text = '''"Ever imagined being a %billionaire% at 21?" "Well, meet %Kylie Jenner%, youngest self-made billionaire by Forbes!" "Born to the famous reality Tv stars, %Kardashians%, Kylie's success isn't any surprise." "But here's the real twist, her wealth isn't just from her family's multibillion-dollar empire." "Turns out, %Kylie Cosmetics%, her beauty brand, is her golden goose, catapulting her to exceptional wealth." "In 2015, she launched the brand with a %lip kit%, which sold out in an unbelievable less than a minute!" "Then she leveraged her massive %social media% following, driving massive sales." "Within just 18 months, Kylie Cosmetics had made a whopping $420 million in retail sales." "Crazy, right? Who says %social media% can't make you a billionaire!" "But wait, there's more! It wasn't all about the lip kits." "Kylie Cosmetics expanded to eyeshadows, eyeliners, concealers and so much more… generating millions, and giving the traditional beauty industry a run for its money." "Then came the jackpot, %Coty Inc.% bought 51% stake in Kylie Cosmetics, valuing it at $1.2 billion!" "Yes, she had controversies, yes, she had challenges. But it's undeniable, Kylie Jenner is a reigning queen of the business world!" "What's stopping you from creating your own %Kylie Cosmetics% in your backyard?"'''

sections = ['Ever imagined being a %billionaire% at 21?', 'Well, meet %Kylie Jenner%, youngest self-made billionaire by Forbes!', "Born to the famous reality Tv stars, %Kardashians%, Kylie's success isn't any surprise.", "But here's the real twist, her wealth isn't just from her family's multibillion-dollar empire.", 'Turns out, %Kylie Cosmetics%, her beauty brand, is her golden goose, catapulting her to exceptional wealth.', 'In 2015, she launched the brand with a %lip kit%, which sold out in an unbelievable less than a minute!', 'Then she leveraged her massive %social media% following, driving massive sales.', 'Within just 18 months, Kylie Cosmetics had made a whopping $420 million in retail sales.', "Crazy, right? Who says %social media% can't make you a billionaire!", "But wait, there's more! It wasn't all about the lip kits.", 'Kylie Cosmetics expanded to eyeshadows, eyeliners, concealers and so much more… generating millions, and giving the traditional beauty industry a run for its money.', 'Then came the jackpot, %Coty Inc.% bought 51% stake in Kylie Cosmetics, valuing it at $1.2 billion!', "Yes, she had controversies, yes, she had challenges. But it's undeniable, Kylie Jenner is a reigning queen of the business world!", "What's stopping you from creating your own %Kylie Cosmetics% in your backyard?"]

audio = "/Users/tassiloar/Desktop/Nexamedia/HEAP/27517259.mp3"
audio_clip = AudioFileClip(audio).set_start(0)

instructions = [['I Kylie Jenner billionaire', [' Kylie Jenner']], ['V Kylie Jenner self-made', [' Kylie Jenner']], ['I Kylie Jenner Kardashians realityTv stars', [' Kardashians']], ['V Kylie Jenner twist wealth', []], ['V Kylie Jenner Cosmetics beauty', [' Kylie Cosmetics']], ['I Kylie Jenner lip kit brand minute', [' lip kit']], ['I Kylie Jenner social media', [' social media']], ['V Kylie Jenner monthsKylieCosmetics sales', []], ['I Kylie Jenner social media billionaire', [' social media']], ['V Kylie Jenner lip kits', []], ['V Kylie Jenner Cosmetics eyeliners', ['Kylie Jenner']], ['I Kylie Jenner Coty Inc jackpot stake', [' Coty Inc']], ['I Kylie Jenner controversies queen business', ['Kylie Jenner']], ['V Kylie Jenner Cosmetics backyard', [' Kylie Cosmetics']]]
spt = [['Ever imagined', '0', '0.600', 1], ['being a', '0.600', '0.900', 3], ['billionaire', '0.900', '1.100', 4], ['at 21', '1.100', 1.8, 6], ['Well meet', 1.8, 2.6, 8], ['Kylie', 2.6, 2.9, 9], ['Jenner', 2.9, '3.100', 10], ['youngest self made', '3.100', 3.5, 11], ['billionaire', 3.5, 4.1000000000000005, 13], ['by', 4.1000000000000005, 4.699999999999999, 14], ['Forbes', 4.699999999999999, 5.0, 15], ['Born to', 5.0, '5.900', 17], ['the famous', '5.900', '6.300', 19], ['reality', '6.300', 6.699999999999999, 20], ['Tv stars', 6.699999999999999, 7.3, 22], ['Kardashians', 7.3, 8.0, 23], ["Kylie's success", 8.0, '8.400', 25], ["isn't", '8.400', '8.600', 26], ['any surprise', '8.600', 9.0, 28], ["But here's", 9.0, '10.100', 30], ['the real', '10.100', '10.300', 32], ['twist', '10.300', '10.500', 33], ['her wealth', '10.500', '11.400', 35], ["isn't", '11.400', '11.600', 36], ['just from her', '11.600', 11.95, 38], ["family's", 11.95, '12.300', 40], ['multibillion', '12.300', 12.9, 41], ['dollar empire', 12.9, 13.299999999999999, 42], ['Turns out', 13.299999999999999, 14.5, 43], ['Kylie', 14.5, 15.0, 45], ['Cosmetics', 15.0, '15.500', 46], ['her beauty', '15.500', '15.900', 48], ['brand', '15.900', '16.200', 49], ['is her', '16.200', 16.65, 50], ['golden', 16.65, '16.900', 52], ['goose', '16.900', '17.100', 53], ['catapulting', '17.100', '18', 54], ['her to', '18', '18.200', 56], ['exceptional', '18.200', '18.600', 57], ['wealth', '18.600', 19.1, 58], ['In 2015', 19.1, '20.300', 60], ['she launched', '20.300', '20.700', 62], ['the brand', '20.700', '21.100', 64], ['with a lip', '21.100', 21.4, 66], ['kit which', 21.4, '22.200', 69], ['sold out', '22.200', '22.600', 71], ['in an', '22.600', '22.800', 73], ['unbelievable', '22.800', '23.300', 74], ['less than', '23.300', '23.600', 76], ['a minute', '23.600', 24.0, 78], ['Then she', 24.0, '24.700', 80], ['leveraged', '24.700', 25.05, 81], ['her massive', 25.05, '25.500', 83], ['social', '25.500', '25.700', 84], ['media', '25.700', '26.100', 85], ['following', '26.100', '26.500', 86], ['driving', '26.500', '27', 87], ['massive', '27', '27.300', 88], ['sales', '27.300', 27.8, 89], ['Within', 27.8, '28.300', 90], ['just 18 months', '28.300', 29.0, 92], ['Kylie', 29.0, 29.400000000000002, 94], ['Cosmetics', 29.400000000000002, 29.799999999999997, 95], ['had made', 29.799999999999997, '30', 97], ['a whopping', '30', 30.099999999999998, 99], ['$420', 30.099999999999998, 31.5, 100], ['million in', 31.5, '31.800', 101], ['retail', '31.800', '32', 102], ['sales', '32', 32.4, 103], ['Crazy', 32.4, '33.300', 104], ['right', '33.300', 33.6, 105], ['Who says', 33.6, '34.100', 107], ['social', '34.100', '34.300', 108], ['media', '34.300', '34.600', 109], ["can't", '34.600', '34.900', 110], ['make you a', '34.900', '35.100', 112], ['billionaire', '35.100', 35.7, 114], ['But wait', 35.7, '36.600', 116], ["there's more", '36.600', 37.2, 117], ["It wasn't", 37.2, '38.200', 120], ['all about the lip', '38.200', 38.5, 123], ['kits', 38.5, 39.2, 125], ['Kylie', 39.2, 39.800000000000004, 126], ['Cosmetics', 39.800000000000004, '40.200', 127], ['expanded to', '40.200', 40.6, 128], ['eyeshadows', 40.6, '41.200', 129], ['eyeliners', '41.200', '41.600', 130], ['concealers', '41.600', '42.100', 131], ['and so', '42.100', '42.300', 133], ['much more…', '42.300', 42.6, 134], ['generating', 42.6, '43.500', 136], ['millions and', '43.500', '44.300', 138], ['giving the', '44.300', '44.700', 139], ['traditional', '44.700', '45.100', 141], ['beauty', '45.100', '45.300', 142], ['industry', '45.300', '45.700', 143], ['a run', '45.700', '46', 145], ['for its money', '46', 46.5, 147], ['Then came', 46.5, 47.4, 150], ['the', 47.4, 47.6, 151], ['jackpot Coty', 47.6, 48.65, 152], ['Inc bought 51 stake in', 48.65, 50.0, 156], ['Kylie', 50.0, 50.300000000000004, 159], ['Cosmetics', 50.300000000000004, '50.700', 160], ['valuing', '50.700', '51.100', 161], ['it at $1.2 billion', '51.100', 52.300000000000004, 167], ['Yes she had', 52.300000000000004, '53.900', 170], ['controversies', '53.900', '54.500', 172], ['yes she had', '54.500', '55.200', 174], ['challenges', '55.200', 55.7, 176], ["But it's", 55.7, '56.100', 178], ['undeniable', '56.100', 56.7, 179], ['Kylie', 56.7, 57.2, 180], ['Jenner', 57.2, '57.300', 181], ['is a reigning', '57.300', '57.800', 184], ['queen', '57.800', '58', 185], ['of the', '58', '58.200', 187], ['business world', '58.200', 58.7, 188], ["What's", 58.7, '59.300', 190], ['stopping', '59.300', '59.500', 191], ['you from', '59.500', 59.85, 193], ['creating your', 59.85, 60.1, 195], ['own', 60.1, 60.5, 196], ['Kylie', 60.5, 60.7, 197], ['Cosmetics', 60.7, '61.200', 198], ['in your backyard', '61.200', '61.600', 201]]

#spt = [['Ever imagined', '0', '0.600', 1], ['being a', '0.600', '0.900', 3], ['billionaire', '0.900', '1.100', 4], ['at 21', '1.100', 1.8, 6], ['Well meet', 1.8, 2.6, 8], ['Kylie', 2.6, 2.9, 9], ['Jenner', 2.9, '3.100', 10], ['youngest self made', '3.100', 3.5, 11], ['billionaire', 3.5, 4.1000000000000005, 13], ['by', 4.1000000000000005, 4.699999999999999, 14], ['Forbes', 4.699999999999999, 5.0, 15], ['Born to', 5.0, '5.900', 17], ['the famous', '5.900', '6.300', 19], ['reality', '6.300', 6.699999999999999, 20], ['Tv stars', 6.699999999999999, 7.3, 22], ['Kardashians', 7.3, 8.0, 23], ["Kylie's success", 8.0, '8.400', 25], ["isn't", '8.400', '8.600', 26], ['any surprise', '8.600', 9.0, 28], ["But here's", 9.0, '10.100', 30], ['the real', '10.100', '10.300', 32], ['twist', '10.300', '10.500', 33], ['her wealth', '10.500', '11.400', 35], ["isn't", '11.400', '11.600', 36], ['just from her', '11.600', 11.95, 38], ["family's", 11.95, '12.300', 40], ['multibillion', '12.300', 12.9, 41], ['dollar empire', 12.9, 13.299999999999999, 42], ['Turns out', 13.299999999999999, 14.5, 43], ['Kylie', 14.5, 15.0, 45], ['Cosmetics', 15.0, '15.500', 46]]


times = find_section_bound(spt, sections)
print(times)

if len(sections) != len(instructions):
  sys.exit() 

count = 0
clips = []
 
s = 0

background = addbackgrund("gradient_motion", [random.randint(0,255),random.randint(0,255),random.randint(0,255)],0, times[len(times)-1][1])

#background = background.subclip(0,10)
#audio_clip = audio_clip.subclip(0,10)

sound_effects = [audio_clip]

audio_clip = CompositeAudioClip(sound_effects)
#clips.append(OpenShot(topic, times[0][1]))

while s < int(len(sections)):  
#while s < 3:  
  clip_temp = createSceneSimple(instructions[s], times[s][0], times[s][1], spt)
  clip_temp = clip_temp.set_start(float(times[s][0]))
  clip_temp = clip_temp.set_end(float(times[s][1]))
  clips.append(clip_temp)
  s+=1



print("XXXXXXXXXX")
print("XXXXXXXXXX")


"""
clip_size = (1080,1920)  
clip_color = (100, 100, 130) 
color_clip = ColorClip(size=clip_size, color=clip_color, duration=audio_clip.duration)
clips.append(color_clip)
"""

print(instructions)
print(times)

text_clips = textVisualnew(spt)

allclips = clips + text_clips

for clip in clips:
  print("    ")
  print(clip.start)
  print(clip.end)
  print(clip.duration)
  print(clip.size[0])
  print(clip.size[1])

final_video = CompositeVideoClip([background]+allclips, size=(1080, 1920))
final_video= final_video.set_audio(audio_clip)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
  
final_video.write_videofile(f"/Users/tassiloar/Desktop/Nexamedia/{formatted_datetime}.mp4", codec="libx264", audio_codec="aac", fps=24, bitrate='3000k')


