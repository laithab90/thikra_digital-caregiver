import pyaudio
import webrtcvad
import collections
import sys
import json
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Annoy
from langchain.embeddings import OpenAIEmbeddings
import openai
import os
from concurrent.futures import ThreadPoolExecutor
import time
import tkinter as tk
import threading
from textblob import TextBlob
from autocorrect import Speller
import nltk
import spacy
import neuralcoref
import os
import pandas as pd
import json
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
# Add NeuralCoref to spaCy's pipe
neuralcoref.add_to_pipe(nlp)
# Initialize the spell checker
spell = Speller(lang='en')
# Download the Punkt tokenizer for sentence splitting
nltk.download('punkt') 
with open(r'chatbot\mofta7.txt', 'r') as file:
    text_data = file.read()
os.environ["OPENAI_API_KEY"] = text_data
openai.api_key = text_data
embeddings_model = OpenAIEmbeddings()
db =  Annoy.load_local("chatbot/my_annoy_index_and_docstore", embeddings=embeddings_model)


class Patient:
    def __init__(self):
        self.sentiment_history = {}  # topics are keys, sentiment scores are values

def replace_pronouns_with_nouns(text):
    doc = nlp(text)
    return doc._.coref_resolved  # this attribute contains the text with pronouns replaced

def correct_spelling_and_lemmatize(text):
    #corrected_text = spell(text)
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text



def extract_topics(text):
    doc = nlp(text)
    # Just using nouns as topics in this simple example
    return [token.text for token in doc if token.pos_ == 'NOUN']

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_conversation(patient, conversation):
    conversation_with_resolved_pronouns = replace_pronouns_with_nouns(conversation)
    corrected_and_lemmatized_conversation = correct_spelling_and_lemmatize(conversation_with_resolved_pronouns)
    return corrected_and_lemmatized_conversation

def handle_patient_response(patient, response):
    # print(f"Patient's response: {response}")
    topics = extract_topics(response)
    # print(f"Identified topics: {topics}")
    sentiment = get_sentiment(response)
    # print(f"Sentiment of response: {sentiment}")

    for topic in topics:
        if topic not in patient.sentiment_history:
            patient.sentiment_history[topic] = []
        patient.sentiment_history[topic].append(sentiment)

    # print(f"Patient's sentiment history: {patient.sentiment_history}")

# Initialize a patient
patient = Patient()

conversation_path = "chatbot/feedbackloop/log_chatgpt_feedbackloop.csv"
sentiment_path = "chatbot/feedbackloop/sentiment_scores.json"
def analyze_conversation_and_save_topics(file_path, json_file_path):
    # If the file doesn't exist, do nothing
    if not os.path.exists(file_path):
        return

    # Load the dataframe from the file
    df = pd.read_csv(file_path, encoding='latin-1')

    # If the JSON file exists, load the existing topics and their scores
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_sentiment_history = json.load(f)
    else:
        existing_sentiment_history = {}

    # Initialize a patient
    patient = Patient()

    # Join all the conversations into a single text, separated by " ||| "
    all_conversations = " ||| ".join(df['patient'])

    # Replace the pronouns in the entire text
    resolved_conversations = process_conversation(patient, all_conversations)

    # Split the text back into individual conversations
    # We assume that the original conversations did not contain the string " ||| "
    resolved_conversations = resolved_conversations.split(" ||| ")

    for conversation in resolved_conversations:
        sentences = nltk.sent_tokenize(conversation)

        for sentence in sentences:
            handle_patient_response(patient, sentence)

    # Get the compound sentiment scores for the topics
    compound_scores = {topic: sum(sentiments)/len(sentiments) for topic, sentiments in patient.sentiment_history.items()}

    # Update the sentiment history for each topic
    for topic, score in compound_scores.items():
        if topic in existing_sentiment_history:
            existing_sentiment_history[topic].append(score)
        else:
            existing_sentiment_history[topic] = [score]

    # Save the sentiment history to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(existing_sentiment_history, f)
analyze_conversation_and_save_topics(conversation_path, sentiment_path)
def get_topics_to_avoid(json_file_path, decay_factor=0.8):
    # If the JSON file doesn't exist, return an empty list
    if not os.path.exists(json_file_path):
        return []

    # Load the sentiment history from the JSON file
    with open(json_file_path, 'r') as f:
        sentiment_history = json.load(f)

    # Calculate the decayed compound scores for the topics
    decayed_scores = {topic: sum(score * (decay_factor ** i) for i, score in enumerate(scores)) 
                      for topic, scores in sentiment_history.items()}

    # Get the topics with a negative decayed compound score
    negative_topics = {topic: score for topic, score in decayed_scores.items() if score < 0}

    # Sort the negative topics in descending order of their decayed compound scores
    topics_to_avoid = sorted(negative_topics.items(), key=lambda item: item[1])

    # Extract only the topics, not their scores
    topics_to_avoid = [topic for topic, score in topics_to_avoid]

    return topics_to_avoid

topics_to_avoid = get_topics_to_avoid(sentiment_path)
if not topics_to_avoid:
    topics_to_avoid = ['none']

# Convert the list of topics to a comma-separated string
topics_str = ', '.join(topics_to_avoid)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

# Include the topics directly in the prompt template
template = f"""Your name is Thikra. You are a highly qualified caregiver in an elderly care facility specializing in dementia. You always stay in your role 
as a caregiver, even when the input is incomplete or ambiguous. You are always talking to a dementia patient. Your role is to help improve 
the mood of the patient through reminiscence therapy. Be proactive and ask questions about the patient's memories.Evoke old memories 
whenever the patient is feeling scared, distressed, confused, or angry.Avoid talking about these topics: {topics_str}.Use the following 
context (delimited by <ctx></ctx>) which is excrepts from book about dementia and reminiscence therapy, and the chat history 
(delimited by <hs></hs>) to form coherent responses to whatever the patient says.:
------
<ctx>
{{context}}
</ctx>
------
<hs>
{{history}}
</hs>
------
{{question}}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],  
    template=template,
)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=70),
    chain_type='stuff',
    retriever=db.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

def monitor(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a dementia patient's monitoring algorithms. Every sentence you see, you return 'safe' or 'dangerous' depending on whether you think that the patient needs urgent medical attention because they might be in pain or sick or in an immediately dangerous situation."},
            {"role": "user", "content": query},
        ],
        max_tokens=15,
        n=1,
        stop=None,
        temperature=0,
    )

    return response.choices[0].message['content'].strip()


# Create a new DataFrame for storing questions and answers
log_chatgpt = pd.DataFrame(columns=['patient', 'caregiver','safety'])


def save_session(new_df, new_df_filename):
    # Save the new DataFrame as a CSV file
    new_df.to_csv(new_df_filename, index=False)

def talk_chatgpt(question):
    # Start the timer
    start_time = time.time()

    # Use a ThreadPoolExecutor to run the model and monitor function in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(qa.run, {"query": question})
        future2 = executor.submit(monitor, question)

    # Get the results
    response = future1.result()
    safety = future2.result()

    # Add the results to the log DataFrame
    log_chatgpt.loc[len(log_chatgpt)] = {'patient': question, 'caregiver': response,  'safety': safety}

    # Stop the timer and calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken to process the question and generate a response: {elapsed_time} seconds")

    return response

from gtts import gTTS
from gtts.tokenizer.pre_processors import abbreviations, end_of_line
import time
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pydub import AudioSegment
from io import BytesIO
import simpleaudio as sa
def transcribe(audio):
    start_time = time.time()

    # Check if the filename already ends with .wav
    if not audio.endswith(".wav"):
        audio_filename_with_extension = audio + '.wav'
        os.rename(audio, audio_filename_with_extension)
    else:
        audio_filename_with_extension = audio
    
    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    response = talk_chatgpt(transcript["text"])
    
    tts = gTTS(response, lang='en', slow=False, pre_processor_funcs=[abbreviations, end_of_line]) 
    
    temp_file = "temp.mp3"
    tts.save(temp_file)
    
    with open(temp_file, "rb") as f:
        mp3_fp = BytesIO(f.read())

    # Load the mp3 data into pydub for playback
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    audio_data = np.array(audio.get_array_of_samples())

    # Start playing the audio in a separate thread
    threading.Thread(target=play_audio, args=(audio_data, audio.channels, audio.sample_width, audio.frame_rate)).start()

    # Stop the timer and calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken to transcribe and respond: {elapsed_time} seconds")

    return response, transcript["text"]  # return the response and the transcript

global audio_playing
audio_playing = False

def play_audio(audio_data, channels, sample_width, frame_rate):
    global audio_playing
    audio_playing = True
    play_obj = sa.play_buffer(audio_data, channels, sample_width, frame_rate)
    play_obj.wait_done()
    audio_playing = False

import collections
import contextlib
import sys
import wave
import webrtcvad
import pyaudio

def record_audio_from_microphone():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
    PADDING_DURATION_MS = 1000  # 1.5 sec
    CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
    NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
    NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)

    vad = webrtcvad.Vad(2)

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     start=False,
                     frames_per_buffer=CHUNK_SIZE)

    stream.start_stream()  # Start the stream

    got_a_sentence = False
    leave = False

    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False

    voiced_frames = []
    while not leave:
        chunk = stream.read(CHUNK_SIZE)
        active = vad.is_speech(chunk, RATE)

        sys.stdout.write('1' if active else '0')

        ring_buffer.append((chunk, active))

        if not triggered:
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.8 * ring_buffer.maxlen:
                sys.stdout.write(' Open ')
                triggered = True
                voiced_frames.extend([f for f, speech in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(chunk)
            ring_buffer.append((chunk, active))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write(' Close ')
                triggered = False
                got_a_sentence = True

        sys.stdout.flush()

        if got_a_sentence:
            data = b''.join([f for f in voiced_frames])
            voiced_frames = []
            ring_buffer.clear()
            got_a_sentence = False
            leave = True

    stream.stop_stream()
    stream.close()

    filename = "output.wav"
    with contextlib.closing(wave.open(filename, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
    
    return filename
# The rest of your script remains the same until here

import tkinter as tk
from tkinter import Canvas, Text, Button, PhotoImage
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\labub\OneDrive\Documents\Data Driven Design\FML\chatbot\build\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Create an instance of the GUI
window = tk.Tk()
window.geometry("1280x720")
window.configure(bg = "#FFFFFF")
background_image = PhotoImage(file=relative_to_assets("image_1.png")) # Replace with your background image path
canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=720,
    width=1280,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)
canvas.create_image(640, 360, image=background_image)  # Place the background image at the center of the canvas


# Display images and text
# Add your image paths
# canvas.create_image(640.0, 360.0, image=PhotoImage(file=relative_to_assets("image_1.png")))
canvas.create_text(574.0, 27.0, anchor="nw", text="thikra", fill="#295282", font=("Sansation Light", 64 * -1))
canvas.create_image(1220.0, 57.0, image=PhotoImage(file=relative_to_assets("image_2.png")))

# Button to terminate the loop
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: window.destroy(),  # This will close the window
    relief="flat",
    bg = "#d7e7eb"
)
button_1.place(x=520.0, y=41.0, width=54.0, height=46.0)

# Text widget for displaying the response
entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(618.5, 364.5, image=entry_image_1)
entry_1 = Text(bd=0, bg="#D7E7EB", fg="#295282", wrap='word', highlightthickness=0, font=("Georgia", 25))
entry_1.place(x=335.0, y=224.0, width=615.0, height=279.0)

def update_response(response):
    # Clear the existing response
    entry_1.delete('1.0', tk.END)
    # Start the typing animation
    type_response(response, 0)

def type_response(response, pos):
    if pos < len(response):
        # Add the next letter to the response
        entry_1.insert(tk.END, response[pos])
        # Scroll to the end of the text widget
        entry_1.see(tk.END)
        # Schedule the next letter to be added after a delay
        window.after(50, type_response, response, pos + 1)  # adjust delay as needed



# Image for safe/dangerous switch
image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
image_5_label = canvas.create_image(1205.0, 75.0, image=image_5)
canvas.tag_raise(image_5_label)  # Bring image_5 to the top
# Initially hide the image_5
canvas.itemconfigure(image_5_label, state='hidden')

image_5_shown = False

def update_image_5_status(status):
    global image_5_shown  # Declare the variable as global so we can modify it
    # If status is 'dangerous', show the image_5
    if status.lower() == 'dangerous':
        print("Status is dangerous, showing image_5")  # Debug print
        canvas.itemconfigure(image_5_label, state='normal')
        image_5_shown = True  # Remember that image_5 has been shown
    # If status is 'safe', hide the image_5 only if it has not been shown before
    elif not image_5_shown:
        print("Status is safe, hiding image_5")  # Debug print
        canvas.itemconfigure(image_5_label, state='hidden')

def main_loop():
    global audio_playing
    while True:
        while audio_playing:  # Wait until previous audio playback is complete
            time.sleep(0.1)
        audio_file = record_audio_from_microphone()
        response, transcript = transcribe(audio_file)
        safety = monitor(transcript)
        print("Transcription Response:", response)
        print("Transcript:", transcript)
        print("Safety:", safety)
        window.after(0, update_response, response)
        window.after(0, update_image_5_status, safety)
        if 'Goodbye' in response or 'goodbye' in response:
            break
    save_session(log_chatgpt, conversation_path)

# Run the main loop in a separate thread
threading.Thread(target=main_loop).start()

# Start the Tkinter event loop
window.mainloop()
