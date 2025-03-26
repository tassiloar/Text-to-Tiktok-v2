
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import random


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
      
      print(words[i])
      if words[i][0] == '%':
        
        term = ""
        xx = 0
        ff = False
        while words[i+xx][len(words[i+xx])-1] != '%':
        
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
# Function to check if the word is a noun
def is_noun(word):
    tagged_word = pos_tag(word_tokenize(word))
    # Check if the first (and only) word's POS tag starts with 'N' (indicating a noun)
    return tagged_word[0][1].startswith('N')



import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Function to check if the input word is a noun
def is_noun_spacy(word):
    doc = nlp(word)
    # Check if the first (and likely only) word's POS tag is a noun
    return doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN"


from moviepy.editor import *
 
def swipe_in_clip(clip, duration=0.2):
  
    pos_dict = [
          lambda t: (min(0, 1080 * (t / duration - 1)), "center"),
          lambda t: (max(0, 1080 * (1 - t / duration)), "center"),
          lambda t: ("center", min(0, 1920 * (t / duration - 1))),
          lambda t: ("center", max(0, 1920 * (1 - t / duration))),
    ]


    return clip.set_position(pos_dict[random.randint(0,len(pos_dict)-1)])


def crossfadein(clip, duration=0.75):
    # Check if the clip has a mask; if not, create an opaque mask for it
    if clip.mask is None:
        clip = clip.set_mask(ColorClip(clip.size, color=1, ismask=True, duration=clip.duration))
    
    # Apply the fadein effect to the mask of the clip
    clip.mask = clip.mask.fx(vfx.fadein, duration)
    return clip


def spin_in(clip, duration=1):

    #clip.set_position( lambda t: (min(0, 1080 * (t / duration - 1)), "center"))
    clip = clip.rotate( lambda t: int(t*100))


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



import cv2

def find_main_figure_position(image_path, threshold_ratio=-0.1):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not loaded properly. Check the file path.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Divide the image into three vertical sections (left, center, right)
    height, width = edges.shape
    left = edges[:, :width // 3]
    center = edges[:, width // 3: 2 * width // 3]
    right = edges[:, 2 * width // 3:]

    # Count the edges in each region
    count_left = cv2.countNonZero(left)
    count_center = cv2.countNonZero(center)
    count_right = cv2.countNonZero(right)

    # Calculate the total count to determine ratios
    total_count = count_left + count_center + count_right

    # Calculate the ratio of each section compared to the total count
    ratio_left = count_left / total_count
    ratio_right = count_right / total_count

    # Apply threshold to decide if the figure is definitively on one side
    if ratio_left > 0.5 + threshold_ratio:
        return "right"
    elif ratio_right > 0.5 + threshold_ratio:
        return "left" 
    else:
        return "center"

import numpy as np

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



def change_color(clip, target_color, new_color, threshold=100):
    def color_filter(image):
        # Ensure the image is a writable copy
        writable_image = np.array(image, copy=True)

        # Calculate the Euclidean distance from the target_color
        diff = np.sqrt(((writable_image - target_color) ** 2).sum(-1))

        # Create a mask for pixels within the threshold of the target_color
        mask = diff < threshold

        # Apply the new color to pixels within the mask
        writable_image[mask] = new_color

        return writable_image

    return clip.fl_image(color_filter)

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


"""
# Usage example
clip = VideoFileClip('/Users/tassiloar/Desktop/Nexamedia/Media Library/Background_Video/bnlmotion.mp4')

target_color = np.array([255, 255, 255])  # Red color in RGB
new_color = np.array([0, 255, 0])     # New color (Green) in RGB
modified_clip = change_color(clip, target_color, new_color)
modified_clip.write_videofile('/Users/tassiloar/Desktop/modified_video.mp4')
"""




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
          background_clip = background_clip.set_duration(end_time-start_time)
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          
        else: #solid
          color[0]-=100
          color[1]-=100
          color[2]-=100
          background_clip = ImageClip("/Users/tassiloar/Desktop/Nexamedia/Media Library/Backgrund template/blank.png")
          background_clip = background_clip.set_duration(end_time-start_time)
          increase_by = np.array(color) 
          background_clip = adjust_rgb_background(background_clip, increase_by)
          

        while background_clip.duration < float(end_time):
          background_clip = CompositeVideoClip([background_clip.set_start(0), background_clip.set_start(background_clip.duration)])
        
        background_clip = background_clip.subclip(float(start_time),float(end_time))
        
        return background_clip
   

import requests
import json

def neets_audio(voice, text):
  
  names = [
    "50-cent", "alex-jones", "anderson-cooper", "andrew-tate", "andrew-yang",
    "angela-merkel", "angie", "anna-kendrick", "anthony-fauci", "antonio-banderas",
    "aoc", "ariana-grande", "arnold-schwarzenegger", "ben-affleck", "ben-shapiro",
    "bernie-sanders", "beyonce", "bill-clinton", "bill-gates", "bill-oreilly",
    "billie-eilish", "cardi-b", "casey-affleck", "charlamagne", "conor-mcgregor",
    "darth-vader", "demi-lovato", "dj-khaled", "donald-trump", "dr-dre",
    "dr-phil", "drake", "dwayne-johnson", "elizabeth-holmes", "ellen-degeneres",
    "elon-musk", "emma-watson", "gilbert-gottfried", "greta-thunberg", "grimes",
    "hillary-clinton", "jason-alexander", "jay-z", "jeff-bezos", "jerry-seinfeld",
    "jim-cramer", "joe-biden", "joe-rogan", "john-cena", "jordan-peterson",
    "justin-bieber", "justin-trudeau", "kamala-harris", "kanye-west", "kardashian",
    "kermit", "kevin-hart", "lex-fridman", "lil-wayne", "mark-zuckerberg",
    "martin-shkreli", "matt-damon", "matthew-mcconaughey", "mike-tyson", "morgan-freeman",
    "patrick-stewart", "paul-mccartney", "pokimane", "prince-harry", "rachel-maddow",
    "robert-downey-jr", "ron-desantis", "sam-altman", "samuel-jackson", "sbf",
    "scarlett-johansson", "sean-hannity", "snoop-dogg", "stephen-hawking", "taylor-swift",
    "tucker-carlson", "tupac", "warren-buffett", "will-smith", "william"
]
  
  if voice.lower() in names:
    response = requests.request(
      method="POST",
      url="https://api.neets.ai/v1/tts",
      headers={
        "Content-Type": "application/json",
        "X-API-Key": "27bd7a12c0a846868a1103ea466657e3"
      },
      json={
        "text": text,
        "voice_id": voice,
        "params": {
          "model": "ar-diff-50k",
          "temperature": 1,
          "diffusion_iterations": 200
        }
      }
    )
    
  elif voice.lower() == "male" or (voice.lower() != "female" and random.randint(1,2)==1):
    response = requests.request(
      method="POST",
      url="https://api.neets.ai/v1/tts",
      headers={
        "Content-Type": "application/json",
        "X-API-Key": "27bd7a12c0a846868a1103ea466657e3"
      },
      json={
        "text": text,
        "voice_id": f"us-male-{random.randint(1,5)}",
        "params": {
          "model": "style-diff-500",
        }
      }
    )
  else:
    response = requests.request(
      method="POST",
      url="https://api.neets.ai/v1/tts",
      headers={
        "Content-Type": "application/json",
        "X-API-Key": "27bd7a12c0a846868a1103ea466657e3"
      },
      json={
        "text": text,
        "voice_id": f"us-female-{random.randint(1,6)}",
        "params": {
          "model": "style-diff-500",
        }
      }
    )
  
  id = 111
  
  with open(f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3", "wb") as f:
    f.write(response.content)
  
  return f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3"
  



"""
#/Users/tassiloar/Desktop/musicgen_out.wav

from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["kanye beat"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)


import scipy

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("/Users/tassiloar/Desktop/musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("/Users/tassiloar/Desktop/astronaut_rides_horse.png")
"""

from openai import OpenAI
import time

client = OpenAI(api_key='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')


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
        "description": "Specific instructions for the script, including sentences to include, vocabulary to use, or tone to use"
      },
      "main_figure": {
        "type": "string",
        "description": "The main entity the video is about"
      },
      "length_min": {
        "type": "integer",
        "description": "Desired length of the video in minutes"
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
        "description": "Whether popup icons should be used"
      },
      "icons_color": {
        "type": "boolean",
        "description": "True for colored popups, false for black and white"
      },
      "captions": {
        "type": "boolean",
        "description": "Whether text captions are present in the video"
      },
      "full_screen_img": {
        "type": "boolean",
        "description": "Whether full-screen images are used"
      },
      "full_screen_vid": {
        "type": "boolean",
        "description": "Whether full-screen videos are used"
      },
      "half_screen_img": {
        "type": "boolean",
        "description": "Whether half-screen images are used"
      },
      "half_screen_vid": {
        "type": "boolean",
        "description": "Whether half-screen videos are used"
      },
      "content_color": {
        "type": "boolean",
        "description": "Whether the main media shown has color or is in black and white"
      },
      "swoosh_sound_effect": {
        "type": "boolean",
        "description": "Whether a swoosh sound effect is used"
      },
      "pop_sound_effect": {
        "type": "boolean",
        "description": "Whether a pop sound effect is used"
      },
      "fade_in_transition": {
        "type": "boolean",
        "description": "Whether a fade-in transition is used"
      },
      "cold_cut_transition": {
        "type": "boolean",
        "description": "Whether a cold cut transition is used"
      },
      "swoosh_transition": {
        "type": "boolean",
        "description": "Whether a swoosh cut transition is used"
      },
      "voice": {
        "type": "string",
        "enum": [
          "male",
          "female",
          "50-cent",
          "alex-jones",
          "anderson-cooper",
          "andrew-tate",
          "andrew-yang",
          "angela-merkel",
          "angie",
          "anna-kendrick",
          "anthony-fauci",
          "antonio-banderas",
          "aoc",
          "ariana-grande",
          "arnold-schwarzenegger",
          "ben-affleck",
          "ben-shapiro",
          "bernie-sanders",
          "beyonce",
          "bill-clinton",
          "bill-gates",
          "bill-oreilly",
          "billie-eilish",
          "cardi-b",
          "casey-affleck",
          "charlamagne",
          "conor-mcgregor",
          "darth-vader",
          "demi-lovato",
          "dj-khaled",
          "donald-trump",
          "dr-dre",
          "dr-phil",
          "drake",
          "dwayne-johnson",
          "elizabeth-holmes",
          "ellen-degeneres",
          "elon-musk",
          "emma-watson",
          "gilbert-gottfried",
          "greta-thunberg",
          "grimes",
          "hillary-clinton",
          "jason-alexander",
          "jay-z",
          "jeff-bezos",
          "jerry-seinfeld",
          "jim-cramer",
          "joe-biden",
          "joe-rogan",
          "john-cena",
          "jordan-peterson",
          "justin-bieber",
          "justin-trudeau",
          "kamala-harris",
          "kanye-west",
          "kardashian",
          "kermit",
          "kevin-hart",
          "lex-fridman",
          "lil-wayne",
          "mark-zuckerberg",
          "martin-shkreli",
          "matt-damon",
          "matthew-mcconaughey",
          "mike-tyson",
          "morgan-freeman",
          "patrick-stewart",
          "paul-mccartney",
          "pokimane",
          "prince-harry",
          "rachel-maddow",
          "robert-downey-jr",
          "ron-desantis",
          "sam-altman",
          "samuel-jackson",
          "sbf",
          "scarlett-johansson",
          "sean-hannity",
          "snoop-dogg",
          "stephen-hawking",
          "taylor-swift",
          "tucker-carlson",
          "tupac",
          "warren-buffett",
          "will-smith",
          "william"
        ],
        "description": "Name of the voice narrator to use"
      }
    },
    "required": [
      "topic",
      "main_figure",
      "text_color",
      "background_graphic",
      "background_color"
    ]
  }
}]
  )
  
  return json.loads(message.choices[0].message.function_call.arguments)
  


instruction = "hello"
print(instruction[2:len(instruction)])


import subprocess
def change_pitch_speed(input_file,length_min):
    
    audio_clip = AudioFileClip(input_file)
    ratio = audio_clip.duration/((length_min*60)+random.randint(2,10))
    
    audio_clip = audio_clip.fx(vfx.speedx, ratio)
    audio_clip.write_audiofile("/Users/tassiloar/Desktop/new.mp3")
    """
    Change the pitch of an audio file using ffmpeg without altering the duration.
    
    Parameters:
    - input_file: Path to the input audio file.
    - output_file: Path to the output audio file with the pitch changed.
    - semitones: Number of semitones by which to change the pitch. Negative values lower the pitch.
    """
    # Construct the ffmpeg command for pitch shifting
    command = [
        'ffmpeg',
        '-i', "/Users/tassiloar/Desktop/new.mp3",
        '-filter_complex', f"rubberband=pitch={2 ** ((-ratio*6) / 12.0)}",
        '-y', input_file
    ]
    
    # Execute the command
    subprocess.run(command, check=True)

def generate_unique_id():
    return random.randint(10000000, 99999999)
  
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
    'Speed': '0.35', # -1.0 to 1.0
    'Pitch': '1', # -0.5 to 1.5
    'TimestampType': 'word', # word or sentence
   #'CallbackUrl': '<URL>', # pinged when ready
  }
)
  
  time.sleep(5)
  data = response.json()
  
  link = data['SynthesisTask']['OutputUri']
  id = generate_unique_id()
  
  response = requests.get(link)
  with open(f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3", 'wb') as audio_file:
      audio_file.write(response.content)
  
  timestamps = data['SynthesisTask']['TimestampsUri']
  response = requests.get(timestamps)
  timestamps_data = response.json()
  
  for segment in timestamps_data:
    times_list.append([segment['word'],segment['start'],segment['end']])
  

  return [f"/Users/tassiloar/Desktop/Nexamedia/HEAP/{id}.mp3",times_list]

text = """%Kanye West%' built his %Yeezy% brand by blending innovation and creativity. Initially known as a Grammy-winning rap artist, West's interest in fashion led to %Nike% collaborations, producing the sought-after %'Air Yeezy' sneakers% in 2009. The marriage of music and fashion proved highly profitable. Yet, it was %West%'s keenness for style autonomy that sparked the birth of %Yeezy% with %Adidas% in 2013."""

text = text.replace('%', '')

print(unreal_speech("x",text))