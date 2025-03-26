from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, \
    concatenate_videoclips, AudioFileClip
import os
import random

def create_video(caption_data):
    # List to hold all processed video clips
    final_clips = []

    # Path to the directory containing the source videos
    content_dir = './TEMP_data/TEMP_content'

    # Initialize previous end time
    previous_end_time = 0

    for index, caption_block in enumerate(caption_data):
        # Load the corresponding video file
        video_path = os.path.join(content_dir, f'{index}.mp4')
        
        if not os.path.isfile(video_path):
            print(f"Video file {video_path} does not exist.")
            continue  # Skip if the video file doesn't exist
        
        # Load the video clip
        video = VideoFileClip(video_path)
        
        #Starts at begginig of first word
        if index == 0:
            block_start_time = 0
        else:
            block_start_time = caption_block[0][1]
        
        block_end_time = caption_block[-1][2]
        block_duration = block_end_time - block_start_time
        
        # Select a random start time within the source video
        max_start_time = max(0, video.duration - block_duration)
        if max_start_time <= 0:
            clip_start_time = 0
        else:
            clip_start_time = random.uniform(0, max_start_time)
        
        # Trim the video clip
        trimmed_clip = video.subclip(clip_start_time, clip_start_time + block_duration)
        
        # Convert to vertical dimensions (e.g., 9:16 aspect ratio)
        vertical_clip = trimmed_clip.resize(width=720, height=1280)
        vertical_clip = vertical_clip.crop(x_center=vertical_clip.w / 2, y_center=vertical_clip.h / 2, width=720, height=1280)
        
        # Create text clips for each caption sequence
        text_clips = []
        for caption in caption_block:
            text, start_time, end_time = caption
            # Calculate the time relative to the trimmed clip
            relative_start = start_time - block_start_time
            relative_end = end_time - block_duration
            relative_end = min(relative_end, block_duration)
            
            txt_clip = TextClip(
                txt=text,
                fontsize=50,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial',
                method='caption',
                size=(vertical_clip.w * 0.8, None),
                align='center'
            ).set_position(('center', 'center')).set_start(relative_start).set_duration(relative_end - relative_start)
            
            text_clips.append(txt_clip)
        
        # Overlay text clips onto the video clip
        composite_clip = CompositeVideoClip([vertical_clip] + text_clips)
        
        # Handle the starting time of the current clip
        if index == 0:
            # If it's the first clip, start at time 0
            composite_clip = composite_clip.set_start(0)
        else:
            # Start at the middle between the end of the last and start of current
            last_end = previous_end_time
            current_start = previous_end_time
            clip_start = (last_end + current_start) / 2
            composite_clip = composite_clip.set_start(clip_start)
        
        # Update previous end time
        previous_end_time = composite_clip.end
        
        # Add to the list of final clips
        final_clips.append(composite_clip)
    
    # Concatenate all the composite clips
    if not final_clips:
        print("No clips were processed. Final video was not created.")
    
    final_video = concatenate_videoclips(final_clips, method='compose')
    
    audio = AudioFileClip('./TEMP_data/TEMP_tts/tts.mp3')
    final_video_with_audio = final_video.set_audio(audio)
    # Export the final video
    output_path = './TEMP_data/output/final_video.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_video_with_audio.write_videofile(output_path, fps=30, codec='mpeg4')
    print(f"Final video saved to {output_path}")

