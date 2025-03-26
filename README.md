# Demo
**Prompt: “Make a 1m video on how Kylie Jenner became the worlds youngest billionaire. Use a gray gradient motion background, captions, and popup pictures. Use a male narator.” ** \\
Result: https://youtube.com/shorts/g6KztCNRliA?feature=share \\
\\
**Prompt: “Make a 1m video on the origin of the paypal mafia. Include this content: https://www.youtube.com/watch?v=s9mczdODqzo, https://pngimg.com/d/paypal_PNG7.png, use transitions, a fitting background and transitions.”** \\
Result: https://youtube.com/shorts/3CTCKqJuB_0?feature=share\\
\\
# ViroClip2 – Text-to-Video Content Generator

**ViroClip2** is a Python-based tool that converts a user’s text description into an engaging short-form video. It leverages OpenAI’s GPT-4 to interpret video creation requests, scrapes the internet for relevant content, and generates a custom voice narration. The result is a video that integrates curated media, dynamic visual effects, and a tailored narration track—all driven by an extensive set of configurable parameters.

---

## How It Works

### 1. Interpreting the User Request

At the core of ViroClip2 is the `ParametersOpenAi` function. Here’s what happens:

- **User Query Input:**  
  The user provides a text prompt describing the desired video. For example:  
  *“Create a video that is 1m long on how Travis Scott became famous with a black and red lined background.”*

- **GPT-4 Parameter Extraction:**  
  The function sends the prompt to OpenAI’s GPT-4 API with a system message that instructs the assistant to interpret the request and invoke the `create_video_instructions` function. This API call returns a JSON object containing all the parameters necessary for video creation.

### 2. Detailed Video Parameter Definitions

The JSON response from GPT-4 includes the following parameters:

- **topic (string):**  
  Main topic of the video. This is used to generate the script and overall narrative.

- **script_specifications (array of strings):**  
  Specific instructions for the script, such as key sentences to include, vocabulary choices, or tone guidelines.

- **restrictive (boolean):**  
  If set to `True`, the video will only include the specific content provided in `script_specifications` without additional material.

- **main_figures (array of strings):**  
  The key entities or characters the video is about. For abstract topics (e.g., “how to find happiness”), the first index should be `"none"`.

- **included_links (array of strings):**  
  A list of URLs for content the user wants included in the video. For example:  
  `["https://www.youtube.com/watch?v=ZrjarkXS0Fo"]`

- **included_images (array of strings):**  
  A list of image prompts (not direct links) to include in the video. For instance:  
  `["drake at the grammys"]`  
  If the prompt is general (e.g., “images of shark tank cast young”), the tool will automatically select relevant images based on recognized names.

- **included_videos (array of strings):**  
  A list of video prompts (not direct links) to include. For example:  
  `["skiing backflip"]`  
  Similar to images, the tool can interpret more general prompts to gather specific video content.

- **length_min (integer):**  
  Desired length of the video in minutes. This parameter is only initialized if the user directly addresses it.

- **text_color (array of integers):**  
  RGB color value for the text overlay. For example:  
  `[255, 255, 255]` for white text.

- **text_outline (array of integers):**  
  RGB color value for the text outline. For example:  
  `[0, 0, 0]` for black outline.

- **background_graphic (string):**  
  Type of background used in the video. Acceptable values are:  
  - `gradient_motion`
  - `lines_motion_black`
  - `lines_motion_white`
  - `paper_motion`
  - `grainy_static`
  - `solid_static`

- **background_color (array of integers):**  
  RGB color value for the background.

- **icons (boolean):**  
  Whether popup icons should be used during the video.

- **icons_color (boolean):**  
  Determines if the popup icons are colored (`True`) or black and white (`False`).

- **captions (boolean):**  
  Whether text captions are present in the video.

- **full_screen (boolean):**  
  Whether full-screen images are used in the video.  
  *(Note: The code further distinguishes between full-screen images and full-screen videos using separate flags internally.)*

- **voice (string):**  
  The narrator's voice, selectable from:  
  - `male`
  - `female`

### 3. Additional Internal Parameters

Beyond the parameters received from GPT-4, the code defines additional internal settings that further customize the video:

- **full_screen_img & full_screen_vid (boolean):**  
  Separate flags for full-screen images and videos.

- **half_screen_img & half_screen_vid (boolean):**  
  Flags to indicate if half-screen media should be used.

- **content_color (boolean):**  
  Determines if the main media content should be displayed in color (`True`) or black and white (`False`).

- **swoosh_sound_effect & pop_sound_effect (boolean):**  
  Control the usage of swoosh and pop-up sound effects during transitions.

- **fade_in_transition, cold_cut_transition, swoosh_transition (boolean):**  
  Flags that determine which transition effects are applied between scenes.

### 4. Voice Generation and Media Scraping

- **Voice Synthesis:**  
  The `unreal_speech` function uses the provided `voice` parameter to generate an audio narration track by submitting the video script to the UnrealSpeech API. It also retrieves timestamps to help synchronize the narration with video scenes.

- **Media Processing:**  
  The code scrapes the internet for media using the provided `included_links`, `included_images`, and `included_videos` parameters. It processes these inputs, labeling each based on its type (e.g., “I” for images, “V” for videos) and ensuring that the video narrative is supported by appropriate visual content.

### 5. Scene Creation and Final Video Composition

- **Scene Instruction Generation:**  
  The `instructions_new` function breaks the generated script into sections and builds a list of detailed scene instructions. These instructions combine media prompts, text overlays, and visual effects.

- **Video Assembly:**  
  Each scene is created using a function (e.g., `createSceneSimple`) that applies the visual effects, transitions, and media based on the instructions. All scenes are then combined using MoviePy’s `CompositeVideoClip`, and the synthesized audio is overlaid to produce the final video.

- **Output:**  
  The final video file is saved with a timestamp-based filename, and the code logs the parameter values, scene instructions, and processing steps for troubleshooting.

---

## How to Use

1. **Set Up the Environment:**
   - Install Python 3.6+.
   - Install required packages:
     ```bash
     pip install requests moviepy
     ```
   - Configure API keys for OpenAI and UnrealSpeech directly in the script.

2. **Provide Your Video Request:**
   - Edit the `input` variable in the script with your text prompt.
   - The prompt should describe the video’s topic, length, styling, and any specific media or voice preferences.

3. **Run the Script:**
   ```bash
   python viroclip2.py
   ```
   - The script will:
     - Parse the user input to extract all video parameters.
     - Scrape the internet for relevant media.
     - Generate a voice narration track and retrieve word timestamps.
     - Split the generated script into sections and create corresponding scene instructions.
     - Assemble the scenes into a final video and save it locally.

4. **Review the Output:**
   - The final video file will be saved with a unique, timestamp-based name.
   - Console logs provide detailed insights into the parameter values and processing flow.

---

## Dependencies

- **Python Libraries:**
  - `requests` – For interacting with APIs.
  - `moviepy` – For video editing and composition.
  - Other standard libraries: `random`, `time`, `datetime`, etc.

- **APIs:**
  - [OpenAI GPT-4](https://openai.com/api/) – For interpreting user queries and extracting video parameters.
  - [UnrealSpeech](https://unrealspeech.com/) – For voice synthesis and narration generation.

