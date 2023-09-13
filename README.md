# Thikra: Caregiver Chatbot with Audio Monitoring

## Overview

"Thikra" is a graduation project aiming to create an advanced chatbot that serves as a caregiver for individuals with dementia. The chatbot leverages various NLP techniques and OpenAI's capabilities to interpret and respond to audio inputs. One of the standout features is the ability to monitor conversations for safety and visually alert the user through a GUI.

## Dependencies

- **pyaudio**: Enables audio I/O.
- **webrtcvad**: Used for Voice Activity Detection.
- **openai**: Integrates OpenAI's functionalities.
- **tkinter**: Constructs the GUI for the application.
- **spaCy**: Used for various NLP tasks.
- **TextBlob**: Determines sentiment.
- **Autocorrect**: Corrects spelling.
- **nltk**: Used for tokenization.
- **neuralcoref**: Enables coreference resolution.

## Key Functionalities

- **Audio Recording**: Records user's voice in real-time.
- **Transcription**: Converts the recorded audio into text for further processing.
- **NLP Tasks**: Includes coreference resolution, topic extraction, and sentiment analysis.
- **Safety Monitoring**: Analyzes the conversation to determine its safety. Unsafe conversations trigger a visual alert on the GUI.
- **Interactive GUI**: Designed using `tkinter`, it provides an interface for the user to interact with the chatbot and receive visual feedback.

## Code Structure

- **Initialization**: Sets up NLP tools, the OpenAI API, and other necessary configurations.
- **Patient Class**: Captures the sentiment history of topics for each patient.
- **Helper Functions**: Functions that perform NLP tasks, audio recording, transcribing, and safety monitoring.
- **GUI Construction**: Builds and updates the graphical interface.
- **Main Loop**: A continuous loop that records audio, processes it, updates the GUI, and monitors for safety.

## Usage

1. Install all dependencies.
2. Ensure you have an OpenAI API key and place it in the designated location.
3. Run the script. The GUI will launch, and the application will start listening for audio input.
4. Speak into the microphone. The chatbot will transcribe, process, respond, and update the GUI accordingly. Any safety concerns will trigger a visual alert.
5. End the conversation with a "Goodbye" to save the session and close the application.

## Conclusion

"Thikra" stands out as a caregiver chatbot that goes beyond text-based interactions. By incorporating audio monitoring and safety alerts, it ensures a higher level of care and attention for individuals with dementia.


![structure](https://github.com/laithab90/thikra_digital-caregiver/assets/95342563/abb7a16e-ad4c-4f63-a451-b8fe9f7cbd83)

![topic modelling](https://github.com/laithab90/thikra_digital-caregiver/assets/95342563/ee95f14b-b01a-4f0f-a57e-8fda0e0c930f)

![timedecay](https://github.com/laithab90/thikra_digital-caregiver/assets/95342563/8b322dac-02fd-4d55-a633-99d0ae2a0665)

![topic modelling](https://github.com/laithab90/thikra_digital-caregiver/assets/95342563/6cb79252-6b88-4113-a704-7bef9f3201e9)
