# install the library (Command for Windows Terminal)
# pip install sounddevice scipy transformers speechbrain
from pydub import AudioSegment
import os
from speechbrain.pretrained import EncoderClassifier

# Function to convert mp3/mp4 to wav
def convert_to_wav(input_file, output_file='output.wav'):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format='wav')
    print(f"Converted {input_file} to {output_file}")
    # return output_file

# Function to analyze if the audio contains human speech

def analyze_human_voice(file_name='output.wav'):
    # Load a pre-trained model for speaker recognition
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    
    # Run the model on the WAV file
    signal = classifier.load_audio(file_name)
    prediction = classifier.classify_batch(signal)
    
    # Get predicted label
    predicted_label = prediction[3]  # This will contain the predicted label information
    print(f"Predicted label: {predicted_label}")
    
    # Return True if human speech is detected, else False
    # This can be customized based on what labels the model returns
    return "human" in predicted_label

# Main process
def process_audio(input_file):
    # Step 1: Convert input mp3/mp4 to wav
    wav_file = convert_to_wav(input_file)
    
    # Step 2: Analyze the converted WAV file for human voice
    is_human = analyze_human_voice(wav_file)
    
    if is_human:
        print(f"{input_file} contains human voice.")
    else:
        print(f"{input_file} does NOT contain human voice.")
        
    return is_human

# Example usage
if __name__ == "__main__":
    input_audio_file = 'example.mp3'  # Replace with your file (mp3 or mp4)
    process_audio(input_audio_file)

