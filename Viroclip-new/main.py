
import use_anthropic
import use_youtube
import format_input
import use_googletts
import use_moviepy

# get_content_list 
# Input: The content items and script
# Function: Creates the final content list
# Output: Final content list
def get_content_list(content_items,script):
    # Clean content list
    cleaned_content_items = format_input.remove_format_prefix(content_items)

    # Get keywords from script for every sentence
    keywords_list = format_input.extract_keywords(script)
    
    # Get final content list by mixing keywords list and content items list
    content_list = format_input.generate_content_list(cleaned_content_items, keywords_list)
    
    return content_list

# download_videos
# Input: Content list of strings and resolution
# Function: Searches for the content and downloads a video
# Output: Null, downloads video to ./TEMP_content
def download_videos(content_list, resolution):
    
    i = 0
    
    for content in content_list:
        video_urls = use_youtube.search_youtube(content)
        use_youtube.download_youtube_video(video_urls, i, resolution)
        i+=1

    
# userInput = "video of kanye west childhood include videos of him as a child and of his mother and his concerts"

# # Get video settings
# res = use_anthropic.get_video_params(userInput)

# # Extract from res
# videoTopic = res[0]
# content_duration = res[1]
# content_items = res[2]

# # Generate script
# script = use_anthropic.generate_script(videoTopic, content_duration)

# # Get final content list 
# content_list = get_content_list(content_items,script)

# num_scenes = len(content_list)

# print("")
# print("SCRIPT")
# print(script)
# print("")
# print("content_duration")
# print(content_duration)
# print("")
# print("content_items")
# print(content_items)
# print("")
# print("content_list")
# print(content_list)

script = """Did you know that Kanye West once lived in China for a year as a child? At just 10 years old, little Kanye found himself immersed in a completely different culture, an experience that would later influence his unique perspective on art and music.

Born in Atlanta but raised in Chicago, Kanye's artistic journey began early. His mother, Donda West, was an English professor who recognized her son's creative potential. She bought him his first sampler when he was only 15, igniting a passion that would change the face of hip-hop.

As a teenager, Kanye sold his beats to local artists, earning a reputation as a prodigy producer. But here's a little-known fact: he was also part of a hip-hop group called the Go-Getters. They never hit it big, but it was here that Kanye honed his skills as both a producer and a rapper.

Despite his growing success as a producer, Kanye's dream of becoming a rapper was often dismissed. Record labels saw him as just a beatmaker, not a performer. But Kanye's determination was unshakeable. He'd even rap with his jaw wired shut after a near-fatal car accident, proving that nothing could stop his rise to stardom."""

content_list = ['Kanye West China At Kanye', 'Kanye West as a child', "Kanye West's concerts", "Kanye West's mother", 'Kanye', 'Kanye West as a child', 'Kanye West as a child', 'Kanye', "Kanye West's mother", 'Kanye', 'Kanye West as a child']


#download_videos(content_list, "720p")

script = "HEllo MY name is tassilo hello. Did you know that Kanye West once lived in China."

sequences_text = use_googletts.text_to_speech_with_timepointing(script, 10)

print(sequences_text)

use_moviepy.create_video(sequences_text)
