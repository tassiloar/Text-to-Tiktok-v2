from openai import OpenAI
import time
from pathlib import Path
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, VideoFileClip, CompositeAudioClip, vfx, ColorClip, concatenate_audioclips
import requests
import moviepy.editor as mp
from PIL import Image
import random
from pydub import AudioSegment
import nltk
from nltk.corpus import cmudict
import os
from datetime import datetime
from pytube import YouTube
from pytube import Search
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
import re
from moviepy.video.fx.all import blackwhite
from moviepy.video.fx.loop import loop
from moviepy.editor import *
import textwrap

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")
nltk.download('cmudict')
d = cmudict.dict()

# FUNCTIONS #########################################################################################

client = OpenAI(api_key='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')

links = []
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
      num*= 2
      return search_and_download_image(strr,path,num) # Return None if no image could be downloaded successfully
    else:
      print("Fail")
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
        print(f"Attempt {attempt+1}: Searching for videos with query '{query}' up to {length} seconds long.")
        s = Search(query)
        videos = [video for video in s.results if video.title not in used_links]
 
        for video in videos:
            time.sleep(2)
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


def generate_unique_id():
    return random.randint(10000000, 99999999)

a = 0
b = 0
c = 0

def createSceneSimple(instruction, start_time, end_time, wordsbase, icons, icons_color, full_screen_img, full_screen_vid, half_screen_img, half_screen_vid, content_color, swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition, main_figures, section): 
    print(instruction)
    global a
    global b
    global c
    global ppp
    
    wordsn = list(wordsbase)

    if a + b + c >=6:
      a = 0
      b = 0
      c = 0
  
    clips = [ColorClip(size=(10,10), color=(0, 0, 0), duration=1)]

    vv = False
    ii = False
    if instruction[0][0] == "V" or instruction[0][0] == "v":
      vv = True
      
      if instruction[0][0] == "V":
        print("A")
        instruction[0] = instruction[0][2:len(instruction[0])]
      
        inst = instruction[0] + " footage"
        
        link = download_video(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
        
        clip = VideoFileClip(link)
      
        clip = clip.volumex(0.2)
      else:
        print("B")
        instruction[0] = instruction[0][2:len(instruction[0])]
      
        inst = instruction[0]
        
        def download_video_link(link, path):
                    yt = (link)
                    print(f"downloading {yt}")
                    unique_id = generate_unique_id() 
                    download_path = f'{path}{unique_id}.%(ext)s'
                    command = ['yt-dlp', '-o', f'{download_path}', f'{yt}']
                    try:
                      subprocess.run(command, check=True)
                    except  Exception as e:
                        print(f"The command '{e.cmd}' failed with return code {e.returncode}")
                        raise RuntimeError("yt-dlp command failed!") from e
                    
                    if not os.path.exists(f'{path}{unique_id}.webm'):
                      raise FileNotFoundError(f"The file {path}{unique_id}.webm does not exist.")
                    
                    print(f"Video downloaded successfully:{download_path}")
                    found_video = True
                    return f'{path}{unique_id}.webm'
        
        try:
          print("C")
          link = download_video_link(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
          
          clip = VideoFileClip(link)
      
          clip = clip.volumex(0.2)
          
        except  Exception as e:
          if main_figures > 0:
            print("D")
            link = download_video(main_figures[random.randint(0,len(main_figures)-1)] + " footage", "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
            
            clip = VideoFileClip(link)
      
            clip = clip.volumex(0.2)
          else:
            print("E")
            
            link = search_and_download_image("I "+section, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
            clip = ImageClip(link)
            
            vv = False
            ii = True
            
      #print(link)
      
      if clip.duration < float(end_time) - float(start_time) -2:
        
        loop_count = int((float(end_time) - float(start_time) -2) / clip.duration) + 1
        
        clip = loop(clip, n=loop_count)
        
      
      if  float(end_time) - float(start_time) -2 < float(clip.duration):
        start = random.randint(0, int(float(clip.duration)) - int(float(end_time)) + int(float(start_time))-2)
              
      else:
        start = 0
      
      clip = clip.subclip(start, start+float(end_time)-float(start_time))
      

      
    else:  
      ii = True
      try:
    
        if instruction[0][0] == "I":
          print("G")
          instruction[0] = instruction[0][2:len(instruction[0])]
        
          inst = instruction[0]
          
          link = search_and_download_image(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
          
          clip = ImageClip(link)
        
        else:
          print("H")
          instruction[0] = instruction[0][2:len(instruction[0])]
        
          inst = instruction[0]
          
          def download_image_link(link, path):
                      response = requests.get(link, stream=True, timeout=10)
                      if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                          id = generate_unique_id()
                          file_path = f"{path}{id}.png"
                          with open(file_path, 'wb') as file:
                              file.write(response.content)
                          if os.path.getsize(file_path) > 0:
                              print(link)
                              return file_path
          
          
          print("I")
          link = download_image_link(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
            
          clip = ImageClip(link)
      
      except  Exception as e:
          print("G")
          if len(main_figures) > 0:
            print("K")
            link = search_and_download_image(main_figures[random.randint(0,len(main_figures)-1)], "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
            
            clip = ImageClip(link)

          else:
            print("L")
            
            link = search_and_download_image("I "+section, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
            clip = ImageClip(link)
            
   
    
    if random.randint(1,3+a)<3 and not (vv and not half_screen_vid) and not (ii and not half_screen_img):
        print("STYLE 1")
        clip = aspectRatio(clip, 1080, (random.randint(4,7)/10)*1920)
        clip = transitions(clip, (2*(1080-clip.size[0]))/3, (1920-clip.size[1])/2 + random.randint(-100,100), float(start_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition)
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
        top_half = transitions(top_half, (2*(1080-clip.size[0]))/3, buffer//3, float(start_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition)
        #bottom_half = bottom_half.set_position(('center',(buffer//3)+top_half.size[1]))
        bottom_half = transitions(bottom_half, (2*(1080-clip.size[0]))/3, (buffer//3)+top_half.size[1], float(start_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition)
        
        clips.append(top_half)
        clips.append(bottom_half)
        b+=2
    elif not (vv and not full_screen_vid) and not (ii and not full_screen_img):
      print("STYLE 3")
      clip = aspectRatio(clip, 1080, 1920)
      clip = transitions(clip, (2*(1080-clip.size[0]))/3, (1920-clip.size[1])/2 , float(start_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition)
      clips.append(clip)
      c+=2
    
    else:
        print("STYLE 1")
        clip = aspectRatio(clip, 1080, (random.randint(4,7)/10)*1920)
        clip = transitions(clip, (2*(1080-clip.size[0]))/3, (1920-clip.size[1])/2 + random.randint(-100,100), float(start_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition)
        clips.append(clip)
        a+=2
    
    if content_color == False:
        clips[len(clips)-1] = blackwhite(clips[len(clips)-1])
        
    if icons == True:
      #SIDE IMAGES
      ff = 0
      while ff < len(instruction[1]) and ff < 2:
        if len(instruction[1][ff]) < 4:
          break
        
        if instruction[1][ff][0] == ' ':
          instruction[1][ff] = instruction[1][ff][1:]
        
        
        sizex = random.randint(600,700)
        x_pos = random.randint(-200,200)
        if random.randint(1,2) == 1:
          y_pos = random.randint(350,600)
        else:
          y_pos = random.randint(-600,-350)
        rotate = random.randint(-20,20)
        starttt = 0.15*ff
        
        
        for wordx in wordsn:
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
                  worddd = worddd.replace("'",'')
                  worddd = worddd.lower()
                  return worddd
            
                wordxx = stardaize(word)
                wordbase = stardaize(instruction[1][ff])
                print(wordxx)
                print(wordbase)
                
                if len(wordxx)>4 and len(wordbase)>4:
                  
                  if wordxx[1:4] == wordbase[1:4]:
                    starttt = -float(start_time) +float(wordx[1]) -0.2

                    
                else:
                  if wordxx == wordbase:
                    starttt = -float(start_time) +float(wordx[1]) -0.2
        

        words = instruction[1][ff].split()
        for item in range(len(words)):
          for item2 in range(len(main_figures)):
            if words[item].lower() in main_figures[item2].lower():
              words[item] = main_figures[item2]
    
    
        instruction[1][ff] = ' '.join(words)
        
        from collections import OrderedDict
        words = instruction[1][ff].split()
        unique_words = list(OrderedDict.fromkeys(words))
        instruction[1][ff] = ' '.join(unique_words)
        
        print(instruction[1][ff])
        print(instruction[1][ff])
        
        is_png = random.randint(1,2)
        
        if is_png == 1:
          inst = instruction[1][ff]+" PNG"
        else:
            if random.randint(1,2) == 1:
                inst = instruction[1][ff]
            else:
                inst = instruction[1][ff]

        link = search_and_download_image(inst, "/Users/tassiloar/Desktop/Nexamedia/HEAP/visual/")
        
        print()
        print(link)
        print()
          
        if not link == None:
          
              
          clip = createimageclip(link, rotate, x_pos, y_pos,sizex,sizex, starttt+float(start_time)+(ff*0.2), float(end_time), swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition,is_png)  
          
          if True:  
            print("EEEEEEEEEEEE")
            clip = clip.set_start(starttt+(ff*0.2))
              
            if icons_color == False:
              clip = blackwhite(clip)
              
            clips.append(clip)
          else:
            print("OOOOOOOO")
          
       

        ff+=1
  
    links.append(link)
    final_clip = CompositeVideoClip(clips, size=(1080, 1920))
    
    return final_clip

  
def createimageclip(link, rotate, x_pos, y_pos, wdith, height, start_time, end_time, swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition,is_png):
  
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
    
  if is_png == 1:
    create_image_with_green_background(link)  
    link = createPng(link)
  print(link)
  print("ADDONS")
      
      
  clip = ImageClip(link)   
      #clip2 = ImageClip(link)    
      
  if random.randint(1,2) == 1 :
    if is_png == 2:
      clip = clip.add_mask().rotate(lambda f: (rotate - f),unit='deg', expand=True) 
    else:
      clip = clip.rotate(lambda f: (rotate - f),unit='deg', expand=True) 
              
  else:
    if is_png == 2:
      clip = clip.add_mask().rotate(lambda f: (rotate + f),unit='deg', expand=True) 
    else:
      clip = clip.rotate(lambda f: (rotate + f),unit='deg', expand=True) 
            
  clip = aspectRatio(clip, wdith, height)
      
      #clip2 = clip2.rotate(lambda f: (rotate + f)/2)
      
      #clip2 = aspectRatio(clip2, wdith+35, height+35)
      
      #clip2 = clip2.set_position((-17 ,-17))
      #clip2 = clip2.fl_image(lambda frame: 255 * np.ones(frame.shape, dtype=np.uint8))
      
      #clip = clip.set_start(float(start_time))
      #clip2 = clip2.set_start(float(start_time))
      
      #clip = CompositeVideoClip([clip2,clip])
      
  clip = transitions(clip, ((1080-clip.size[0])/2) + x_pos, ((1920-clip.size[0])/2) +y_pos, start_time, swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition,True)
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


def textVisualnew(words,  text_color, text_outline):
    
    hex_color_text = "#{:02x}{:02x}{:02x}".format(text_color[0], text_color[1], text_color[2])
    hex_color_outline = "#{:02x}{:02x}{:02x}".format(text_outline[0], text_outline[1], text_outline[2])
    
    texts = [];
    
    i = 0
    
    while i < len(words):

        if len(words[i][0]) > 0:
          text_clip1 = TextClip(textwrap.fill(words[i][0],16), 
                                  fontsize=65, 
                                  color=hex_color_text, 
                                  font="Arial-bold")
          
          text_clip_border = TextClip(textwrap.fill(words[i][0],16), 
                                  fontsize=65, 
                                  color=hex_color_outline, 
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
    
    print(words)
    print(setions)
    
    def fomrat(stri):
      stri = stri.replace(',','')
      stri = stri.replace(".","")
      stri = stri.replace('!','')
      stri = stri.replace('?','')
      stri = stri.replace('"','')
      stri = stri.replace("%","")
      stri = stri.replace("#","")
      stri = stri.replace(',','')
      stri = stri.replace(".","")
      stri = stri.replace('!','')
      stri = stri.replace('?','')
      stri = stri.replace('-','')
      stri = stri.replace(';','')
      stri = stri.replace('\n','')
      stri = stri.lower()
      return stri
  
    section_times = []
    
    words_split = list(words)
    
    item = 0
    while item < len(words):
        words[item][0] = fomrat(words[item][0])
        item+=1

    #print("")
    #print(words_split)
    
    for section in setions:
      start = -1
      end = -1

      setion_split = section.split()
      
      item = 0
      while item < len(setion_split):
            setion_split[item] = fomrat(setion_split[item])
            item+=1
        
      #print("")
      #print(setion_split)

      i = 0
      while i < len(words_split)-2:
        
        """
        print(" ")
        print(words_split[i][0])
        print(words_split[i+1][0])
        print(words_split[i+2][0])
        print(" ")
        print(setion_split[len(setion_split)-3])
        print(setion_split[len(setion_split)-2])
        print(setion_split[len(setion_split)-1])
        print(" ")
        print(words_split[i][0] == setion_split[len(setion_split)-3])
        print(words_split[i+1][0] == setion_split[len(setion_split)-2])
        print(words_split[i+2][0] == setion_split[len(setion_split)-1])
        print(" ")
       
        """
        if words_split[i][0] == setion_split[0] and words_split[i+1][0] == setion_split[1] and words_split[i+2][0] == setion_split[2]:
          start = float(words_split[i][1])
          #print("weuidhewdui")
        
        if words_split[i][0] == setion_split[len(setion_split)-3] and words_split[i+1][0] == setion_split[len(setion_split)-2] and words_split[i+2][0] == setion_split[len(setion_split)-1]:
          end = float(words_split[i+2][2])
          #print("3e24efr")
        
        i+=1
        
     
      section_times.append([start,end])
    x = 0
    
    section_times[0][0] = 0
    section_times[len(section_times)-1][1] = words_split[len(words_split)-1][2]
    
    print("OLD")
    print(section_times)
    print("")
    
    while x < len(section_times)-1:
      
      if int(section_times[x][1]) == -1 and int(section_times[x+1][0]) != -1 :
        section_times[x][1] = float(section_times[x+1][0])
      elif int(section_times[x+1][0]) == -1 and int(section_times[x][1]) != -1:
        section_times[x+1][0] = float(section_times[x][1])
      elif float(section_times[x][1]) == -1 and float(section_times[x+1][0])== -1:
        avg = (float(section_times[x][0]) + float(section_times[x+1][1]))/2
        section_times[x][1] = avg
        section_times[x+1][0] = avg
      
      x+=1
    
    x = 0
    while x < len(section_times)-1:
      section_times[x][1] = section_times[x+1][0]
      x += 1
    
    return section_times


# Function to check if the input word is a noun
def is_noun_spacy(word):
    doc = nlp(word)
    # Check if the first (and likely only) word's POS tag is a noun
    return doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN"

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

  
def transitions(clip, x_pos, y_pos, start, swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition, ispopup = False):

    if ispopup == True:
      mov = 0
    else:
      mov = 0
    
    
    duration_swipe = 0.15
    pos_dict = [
            lambda t: (min(0, 1080 * (t / duration_swipe - 1))+x_pos+(t*mov), y_pos),
            lambda t: (max(0, 1080 * (1 - t / duration_swipe)+x_pos)+(t*mov), y_pos),
            lambda t: (x_pos+(t*mov), min(0, 1920 * (t / duration_swipe - 1))+y_pos),
            lambda t: (x_pos+(t*mov), max(0, 1920 * (1 - t / duration_swipe))+y_pos),
      ]
    
    x = random.randint(1,4)
    if (x <=2 or ispopup)and swoosh_transition:
      clip = clip.set_position(pos_dict[random.randint(0,len(pos_dict)-1)])
      if swoosh_sound_effect == True:
        if random.randint(1,1) == 1:
          video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/wooosh1.mp4")
        else:
          video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/woosh2.mp4")
        
        
        sound_effects.append(video.audio.set_start(start).volumex(0.04))
        
      
      
    elif x == 3 and fade_in_transition:
      if clip.mask is None:
            clip = clip.set_mask(ColorClip(clip.size, color=1, ismask=True, duration=clip.duration))

      clip.mask = clip.mask.fx(vfx.fadein, 0.75)
      
      clip = clip.set_position(lambda t:(x_pos+(t*mov),y_pos))
    
    elif x == 4 and cold_cut_transition:
      if pop_sound_effect == True:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/pop.mp4")
        sound_effects.append(video.audio.set_start(start).volumex(0.02))
        
      clip = clip.set_position(lambda t:(x_pos+(t*mov),y_pos))

    else:
      if pop_sound_effect == True:
        video = VideoFileClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Soun effects/pop.mp4")
        sound_effects.append(video.audio.set_start(start).volumex(0.02))
        
      clip = clip.set_position(lambda t:(x_pos+(t*mov),y_pos))
    return clip

def ParametersOpenAi(query):


  message = client.chat.completions.create(
    model = "gpt-4",
    messages = [{"role": "system", "content": "Your role is to interpret and fulfill user requests for video creation. Please examine each request attentively and invoke the create_video_instructions function with the correct parameters as specified. Should the user omit details for a required parameter, use your judgment to select an appropriate value. This choice should align with the video's overall theme, guided by the parameter's description and any related information provided. Your objective is to ensure that all necessary information is thoughtfully considered to produce a cohesive and on-target video."},
                {"role": "user", "content":query}],
    functions = [{
  "name": "create_video_instructions",
  "description": "Define the variables of a video",
  "parameters": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Main topic of the video, which will be used to generate the script"
      },
      "script_specifications": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Specific instructions for the script. Such as sentences to include, vocabulary to use, or tone to use"
      },"restrictive": {
        "type": "boolean",
        "description": "If there is an item added to script_specifications, only make restrictive True is the user explicitly stated they only want the content they provided to be the one used in the video"
      },
      "main_figures": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "The main entities the video is about. If the topic of the video is on an abstract concept such as 'how to find happiness', enter 'none' as the first index."
      },
       "included_links": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "This is a list of links of content that the user wants included in the video. For example: 'https://www.youtube.com/watch?v=ZrjarkXS0Fo'"
       },
      "included_images": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "This is a list of image prompts (not links) that the user wants included in the video. For example: 'drake at the grammys'. If the user gives a more general prompt such as 'images of shark tank cast young', it is your job to know the cast of shark tank and input ''name of cast member' young' as an item in the array for each cast member"
      },
       "included_videos": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "This is a list of video prompts (not links) that the user wants included in the video. For example: 'skiing backflip'. If the user gives a more general prompt such as 'videos of shark tank cast young', it is your job to know the cast of shark tank and input ''name of cast member' young' as an item in the array for each cast member"
      },
      "length_min": {
        "type": "integer",
        "description": "Desired length of the video in minutes. Do not initialize unless directly adressed by user."
      },
      "text_color": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "RGB color value for the text"
      },
      "text_outline": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "RGB color value for the text outline"
      },
      "background_graphic": {
        "type": "string",
        "enum": [
          "gradient_motion",
          "lines_motion_black",
          "lines_motion_white",
          "paper_motion",
          "grainy_static",
          "solid_static"
        ],
        "description": "Type of background used"
      },
      "background_color": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "description": "RGB color value for the background"
      },
      "icons": {
        "type": "boolean",
        "description": "Whether popup icons should be used. Do not initialize unless directly adressed by user."
      },
      "icons_color": {
        "type": "boolean",
        "description": "True for colored popups, false for black and white. Do not initialize unless directly adressed by user."
      },
      "captions": {
        "type": "boolean",
        "description": "Whether text captions are present in the video. Do not initialize unless directly adressed by user."
      },
      "full_screen": {
        "type": "boolean",
        "description": "Whether full-screen images are used. Do not initialize unless directly adressed by user."
      },
      "voice": {
        "type": "string",
        "enum": [
          "male",
          "female",
        ],
        "description": "Name of the voice narrator to use. Do not initialize unless directly adressed by user."
      }
    },
    "required": [
      "topic",
      "main_figure",
      "background_color"
    ]
  }
}]
  )
  
  return json.loads(message.choices[0].message.function_call.arguments)

def instructions_new(sections, main_figures, included_media):
  
  instructions = []
  
  last = ""
  
  if random.randint(1,2) == 1:
    last = "I "
  else:
    last = "V "
  
  for i in sections:
    instructions.append([last,[]])
    if  last == "I ":
      last = "V "
    else:
      last = "I "
  
  for i in instructions:
    if random.randint(1,4) == 1:
      if i[0] == "I ":
        i[0] = "V "
      else:
        i[0] = "I "
  
  cur = 0
  
  while cur < len(sections):
    
    cur_let = 0
    
    adding = False
    
    cur_prompt = ""
    cur_word = ""
    popups = []
    
    while cur_let < len(sections[cur]):
      
      if sections[cur][cur_let] == '%' and  adding == False:
        
        adding = True
      
      elif sections[cur][cur_let] == '%' and  adding == True:
        
        adding = False
        cur_prompt += cur_word+" "
        popups.append(cur_word)
        cur_word = ""
      
      if sections[cur][cur_let] != '%' and adding == True:
        
        cur_word += sections[cur][cur_let]
      
      cur_let+=1
    
    #replace with whole name
    words = cur_prompt.split()
    for item in range(len(words)):
      for item2 in range(len(main_figures)):
        if words[item].lower() in main_figures[item2].lower():
          words[item] = main_figures[item2]
    
    cur_prompt = ' '.join(words)

    #remove dups
    from collections import OrderedDict
    words = cur_prompt.split()
    unique_words = list(OrderedDict.fromkeys(words))
    cur_prompt = ' '.join(unique_words)
    
    
    cur_prompt = ' '.join(unique_words)
    
    instructions[cur][0]+=cur_prompt 
    instructions[cur][1] = popups
    
    cur+=1
  
  for instruction in range(len(instructions)):
    if len(instructions[instruction][0]) < 6 or random.randint(1,len(included_media)+2)>2:
      
      if len(included_media) > 0:
          
        num = random.randint(0,len(included_media)-1)
        instructions[instruction][0] = included_media[num]
        if included_media[num][0] == 'i':
          included_media.pop(num)
          
      else:
        if len(main_figures) == 1:
          instructions[instruction][0] += main_figures[0]
        elif len(main_figures) == 0:
          instructions[instruction][0] == "I "+ sections[instruction]
        
        else:
          instructions[instruction][0] += main_figures[random.randint(0,len(main_figures)-1)]
  
    
  return instructions
            
def unreal_speech(voice, text):
  
  times_list = []
  
  response = requests.post(
  'https://api.v6.unrealspeech.com/synthesisTasks',
  headers = {
    'Authorization' : 'Bearer uMmeWM54Az3soxIFyDTQyPdesLtUUmYFtIYRUzW71PxBAa5g1r7q2H'
  },
  json = {
    'Text': text, # Up to 500,000 characters
    'VoiceId': 'Scarlett', # Dan, Will, Scarlett, Liv, Amy
    'Bitrate': '192k', # 320k, 256k, 192k, ...
    'Speed': '0.45', # -1.0 to 1.0
    'Pitch': '1', # -0.5 to 1.5
    'TimestampType': 'word', # word or sentence
   #'CallbackUrl': '<URL>', # pinged when ready
  }
)
  sleeper = 5
  time.sleep(sleeper)
  
  while True:
    data = response.json()
    link = data['SynthesisTask']['OutputUri']
    timestamps = data['SynthesisTask']['TimestampsUri']
    try:
      
      print(sleeper)
   
      print(link)
      id = generate_unique_id()
      
      response = requests.get(link)
      with open(f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3", 'wb') as audio_file:
          audio_file.write(response.content)

      
      print(timestamps)
      responsex = requests.get(timestamps)
      timestamps_data = responsex.json()
      
      for segment in timestamps_data:
        times_list.append([segment['word'],segment['start'],segment['end']])
      break
    except Exception as e:
      time.sleep(sleeper)
      sleeper*= 2
      

  return [f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3",times_list]


    
  
input = "Create a video that is 1m long on how the travis scott became famous use a black and red lined background."


topic = ""
#This is the main topic of the video, this topic will be used to generate teh script of the video

script_spesifications = []
#This is spesific instructions for the script, ti smight included sentences or vocabulary to include, or a spesific tone to use

main_figures = ""
#This is where you input the main entity the user wants the video to be about

length_min = 1
#This is the length in minutes that the user want the video to be 

text_color = [255,255,255]
#This is the color of the text in RGB value

text_outline = [0,0,0]
#This is the color of the text outline in RGB value

backgroun_graphic = "gradient_motion"
#This is the type of background used 
# It must be one of the following: "gradient_motion", "lines_motion_black", "lines_motion_white", "paper_motion", "grainy_static", "solid_static"

background_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
#This is the color of the background used in RGB value

icons = True
#This is wether popup icons should be used

icons_color = True
#True = the popus have color, False = they are in black and white

captions = True
#True = text captions present in video, False =  text captions not present in video

full_screen_img = True
#True = full screen images are used, False = not used
full_screen_vid = True
#True = full screen videos are used, False = not used
half_screen_img = True
#True = half screen images are used, False = not used
half_screen_vid = True
#True = falf screen videos are used, False = not used

content_color = True
#True = the main media shown has color, False = they are in black and white

swoosh_sound_effect = True
#True = swoosh sound effect is used, False = not used
pop_sound_effect = True
#True = pop up sound effect is used, False = not used

fade_in_transition = True
#True = fade in transition is used, False = not used
cold_cut_transition = True
#True = cold cut transition is used, False = not used
swoosh_transition = True
#True = shwoosh cut transition is used, False = not used

voice = "--"

included_images = []
included_videos = []
included_links = []


restrictive = False

Params = ParametersOpenAi(input)

print(Params)

if 'topic' in Params:
  topic = Params['topic']

if 'script_specifications' in Params:
  script_spesifications = Params['script_specifications']
  
if 'main_figures' in Params:
  main_figures = Params['main_figures']
  
if 'length_min' in Params:
  length_min = Params['length_min']
  
if 'text_color' in Params:
  text_color = Params['text_color']
  
if 'text_outline' in Params:
  text_color = Params['text_outline']
  
if 'background_graphic' in Params:
  background_graphic = Params['background_graphic']
  
if 'background_color' in Params:
  background_color = Params['background_color']
  
if 'icons' in Params:
  icons = Params['icons']
  
if 'icons_color' in Params:
  icons_color = Params['icons_color']
  
if 'captions' in Params:
  captions = Params['captions']

if 'full_screen_img' in Params:
  full_screen_img = Params['full_screen_img']

if 'full_screen_vid' in Params:
  full_screen_vid = Params['full_screen_vid']

if 'half_screen_img' in Params:
  half_screen_img = Params['half_screen_img']

if 'half_screen_vid' in Params:
  half_screen_vid = Params['half_screen_vid']
  
if 'content_color' in Params:
  content_color = Params['content_color']
  
if 'swoosh_sound_effect' in Params:
  swoosh_sound_effect = Params['swoosh_sound_effect']
  
if 'pop_sound_effect' in Params:
  pop_sound_effect = Params['pop_sound_effect']
  
if 'fade_in_transition' in Params:
  fade_in_transition = Params['fade_in_transition']
  
if 'cold_cut_transition' in Params:
  cold_cut_transition = Params['cold_cut_transition']
  
if 'swoosh_transition' in Params:
  swoosh_transition = Params['swoosh_transition']
  
if 'voice' in Params:
  voice = Params['voice']

if 'included_images'  in Params:
  included_images = Params['included_images']

if 'included_videos'  in Params:
  included_videos = Params['included_videos']

if 'included_links'  in Params:
  included_links = Params['included_links']

if 'restrictive'  in Params:
  restrictive = Params['restrictive']


included_content = []

print(main_figures)

qq = 0
while qq < len(included_links):
  if included_links[qq].endswith("png") or included_links[qq].endswith("jpg") or included_links[qq].endswith("jpeg"):
    included_links[qq] = "i "+included_links[qq]
  elif "youtube" in included_links[qq]:
    included_links[qq] = "v "+included_links[qq]
  else:
    included_links.pop(qq)
    qq-=1
  
  qq+=1

for item in included_links:
  included_content.append(item)
for item in included_videos:
  included_content.append("V "+item)
for item in included_images:
  included_content.append("I "+item)

specs = ""
for item in script_spesifications:
  specs+=item + ". "

if len(script_spesifications)>0:
 specs = "Spesifications: "+ specs

text = OpenAIAssistant("asst_7zYo27j8yxn6xLBdcM5rzAJr", f"Follow your instructions to create a new text on {topic} between {length_min*210} and {length_min*230} words long. {specs}")

print("text")
print(text)
print()

sections = text.split(". ")

#sections = ['Sure', "Let's dive in, shall we? \n\nWelcome back to another video on your favorite rapper's history! Today, we shall focus on the childhood days of %Playboy Carti%", 'Born as %Jordan Terrell Carter% on September 13, 1996, Playboy Carti spent his early days in %South Atlanta%, the city that boasts of producing world-renowned hip hop artists', 'Little did people know, this high school dropout had melodies running in his veins', '\n\nWhile he was still a rugrat, Carti fell in love with basketball', 'Shooting hoops was his happy place', 'Yet, a young Carti was destined to play a different game of life', '\n\nThis sassy superstar started rapping at an astonishingly young age of 5, outclassing all kids his age with his smooth flow and swaggering persona', 'At 15, Carti ditched textbooks for the dope beat of drum kits and made his music group, %+VLONE Thugz+%', "\n\nBut the world didn't get their %Playboy% overnight", 'His shopping habits at thrift shops quenched his curiosity for unique fashion, helping him build an off-beat aesthetic and giving birth to the image we now associate with him', "Cricket Club was his go-to thrift shop, and the word was his stage.\n\nSo, that's how %Playboy Carti% went from a regular schoolboy to an up-and-coming rapper shaking up the hip hop scene! Stay tuned for more spicy tales of your favorite artists!"]

print("sections")
print(sections)
print()

if not restrictive == True:
    instructions = instructions_new(sections,main_figures, included_content)
    print(instructions)
else:
    instructions = []
    for item in sections:
        num = included_content[random.randint(0,len(included_content)-1)]
        instructions.append(included_content[num])
        if included_content[num][0] == 'i':
          included_content.pop(num)

instructionscopy = list(instructions)




text = text.replace('%','')

sleeper = 2

while True:
  try:
    audio_response = unreal_speech(voice, text)
    break
  except Exception as e:
    time.sleep(sleeper)
    sleeper*= 2

audio_clip = AudioFileClip(audio_response[0]).set_start(0)
spt = audio_response[1]

print("audio")
print(audio_response[0])
print()


print("spt")
print(spt)
print()


times = find_section_bound(spt, sections)

print("times")
print(times)
print()




count = 0
clips = []
 
s = 0

background = addbackgrund(background_graphic, background_color,0, times[len(times)-1][1])


sound_effects = [audio_clip]

audio_clip = CompositeAudioClip(sound_effects)

while s < int(len(sections)):  
  clip_temp = createSceneSimple(instructions[s], times[s][0], times[s][1], spt, icons, icons_color, full_screen_img, full_screen_vid, half_screen_img, half_screen_vid, content_color, swoosh_sound_effect, pop_sound_effect, fade_in_transition, cold_cut_transition, swoosh_transition, main_figures, sections[s])
  clip_temp = clip_temp.set_start(float(times[s][0]))
  clip_temp = clip_temp.set_end(float(times[s][1]))
  clips.append(clip_temp)
  s+=1



print("XXXXXXXXXX")
print("XXXXXXXXXX")

print("")
print("Text")
print(text)

print("")
print("Instructions original")
print(instructionscopy)

print("")
print("Instructions")
print(instructions)

print("")
print("Parameters")
print(Params)

print("")
print("Section bound")
print(times)

print("")
print("USED LINKS")
print(links)


if captions == True:
  text_clips = textVisualnew(spt, text_color, text_outline)

  allclips = clips + text_clips
else:
  allclips = clips


final_video = CompositeVideoClip([background]+allclips, size=(1080, 1920))
final_video= final_video.set_audio(audio_clip)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
  
final_video.write_videofile(f"/Users/tassiloar/Desktop/Nexamedia/{formatted_datetime}.mp4", codec="libx264", audio_codec="aac", fps=24, bitrate='3000k')

