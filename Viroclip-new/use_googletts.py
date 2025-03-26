# Uses google authentication key

import sys
import re

import global_var

from google.cloud.texttospeech_v1beta1 import VoiceSelectionParams, \
    AudioConfig, AudioEncoding, SynthesizeSpeechRequest, \
        SynthesisInput, TextToSpeechClient


api_key = global_var.GOOGLE_API_KEY


client = TextToSpeechClient()

import re
import sys
from google.cloud.texttospeech_v1beta1 import VoiceSelectionParams, \
    AudioConfig, AudioEncoding, SynthesizeSpeechRequest, \
    SynthesisInput, TextToSpeechClient

client = TextToSpeechClient()

def add_marks_to_text(text):
    # Use regex to split text into sentences more accurately
    sentence_endings = re.compile(r'([.!?])')
    sentence_parts = sentence_endings.split(text)
    sentences = [''.join(i) for i in zip(sentence_parts[0::2], \
        sentence_parts[1::2])]

    marked_text = []  # Store the resulting marked text
    word_list = []    # List to store word data

    # Iterate over each sentence
    for sentence_index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Use regex to split sentence into words while keeping 
        # contractions intact
        words = re.findall(r"\b[\w']+\b", sentence)

        marked_sentence = []
        for word_index, word in enumerate(words):
            mark_name = f"{sentence_index}.{word_index}"
            marked_word = f'<mark name="{mark_name}"/>{word}'
            marked_sentence.append(marked_word)

            # Add to word_list
            word_list.append({
                'word': word,
                'sentence_index': sentence_index,
                'word_index': word_index,
                'mark_name': mark_name
            })

        # Join the marked words into a full marked sentence
        marked_text.append(' '.join(marked_sentence))

    # Join all sentences with spaces
    ssml_text = ' '.join(marked_text)
    return ssml_text, word_list

def text_to_speech_with_timepointing(text, max_sequence_size):
    ssml_text, word_list = add_marks_to_text(text)

    request = SynthesizeSpeechRequest(
        input=SynthesisInput(ssml="<speak>" + ssml_text + "</speak>"),
        voice=VoiceSelectionParams(
            language_code='en-US',
            name='en-US-Polyglot-1',
            ssml_gender='MALE'
        ),
        audio_config=AudioConfig(audio_encoding=AudioEncoding.MP3,
                                 speaking_rate=1.0),
        enable_time_pointing=[SynthesizeSpeechRequest.TimepointType.SSML_MARK]
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(request=request)

    # The audio content is returned as binary data (bytes)
    audio_content = response.audio_content

    # Check if audio content is returned correctly
    if audio_content:
        print("\nAudio content received successfully.")
    else:
        print("\nERROR No audio content received. Check your request.\n")
        sys.exit(1)

    # Write the binary MP3 data to an MP3 file
    try:
        with open('./TEMP_data/TEMP_tts/tts.mp3', "wb") as out_file:
            out_file.write(audio_content)
        print(f"Audio content written to ./TEMP_data/TEMP_tts/tts.mp3")
    except Exception as e:
        print(f"\nERROR Failed to write audio file: \n{e}")
        sys.exit(1)

    # Get the list of timepoints
    timepoints = list(response.timepoints)

    # Process the text and timepoints to get the desired sequences
    sequences_text = process_text_and_timepoints(word_list, timepoints, \
        max_sequence_size)

    return sequences_text

def process_text_and_timepoints(word_list, timepoints, max_sequence_size):
    # Build a dictionary to map mark names to timepoints
    mark_to_time = {tp.mark_name: tp.time_seconds for tp in timepoints}

    # Organize words by sentences
    sentences = {}
    for word_info in word_list:
        sentence_index = word_info['sentence_index']
        if sentence_index not in sentences:
            sentences[sentence_index] = []
        sentences[sentence_index].append(word_info)

    # Initialize the outer array to hold sentences
    result = []

    # Process each sentence
    for sentence_index in sorted(sentences.keys()):
        word_infos = sentences[sentence_index]

        # Initialize variables for building sequences
        sequences = []
        current_sequence = []
        current_sequence_length = 0
        start_time = None
        end_time = None

        for word_info in word_infos:
            # Remove punctuation from the word
            word_clean = re.sub(r'[^\w\s\']', '', word_info['word'])
            mark_name = word_info['mark_name']
            word_time = mark_to_time.get(mark_name)

            if word_time is None:
                # Skip if no timepoint is found for this word
                continue

            word_length = len(word_clean)

            if not current_sequence:
                # Start a new sequence
                current_sequence = [word_clean]
                current_sequence_length = word_length
                start_time = word_time
                end_time = word_time
            else:
                # Check if we need to apply the max_sequence_size limit
                if current_sequence_length + word_length > max_sequence_size \
                    and len(current_sequence) > 1:
                    # Finish current sequence
                    sequences.append([
                        ' '.join(current_sequence),
                        start_time,
                        end_time
                    ])
                    # Start a new sequence with the current word
                    current_sequence = [word_clean]
                    current_sequence_length = word_length
                    start_time = word_time
                    end_time = word_time
                else:
                    # Add the word to current_sequence
                    current_sequence.append(word_clean)
                    current_sequence_length += word_length
                    # Update end time to current word's time
                    end_time = word_time  

        # Add any remaining words in the current sequence
        if current_sequence:
            sequences.append([
                ' '.join(current_sequence),
                start_time,
                end_time
            ])

        # Append the sequences of this sentence to the result
        result.append(sequences)

    return result

