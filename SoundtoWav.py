import speech_recognition as sr
from pydub import AudioSegment
import os

def process_audio(audio_file, output_dir):
    """
    Processes an audio file and saves it to the output directory if it contains human voice.

    Args:
        audio_file (str): Path to the audio file.
        output_dir (str): Path to the output directory.
    """
    # Determine file format based on extension
    file_extension = os.path.splitext(audio_file)[1].lower()
    audio_format = "mp3" if file_extension == ".mp3" else "mp4"

    # Load audio using the appropriate method
    try:
        sound = AudioSegment.from_file(audio_file, format=audio_format)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return  # Exit the function if loading fails

    temp_wav_file = "temp.wav"  # Use a temporary file for processing
    sound.export(temp_wav_file, format="wav")

    recognizer = sr.Recognizer()

    with sr.AudioFile(temp_wav_file) as source:
        audio = recognizer.record(source)

    try:
        recognizer.recognize_google(audio)  # Attempt speech recognition

        # If speech is recognized, save the original file to the output directory
        output_file = os.path.join(output_dir, os.path.basename(audio_file))
        os.rename(audio_file, output_file)  # Move the original file to the output directory
        print(f"Human voice detected in {audio_file}. Saved to {output_file}")

    except sr.UnknownValueError:
        print(f"No human voice detected in {audio_file}. File rejected.")
        os.remove(temp_wav_file)  # Remove the temporary WAV file
    except sr.RequestError:
        print(f"Could not request results from Google Speech Recognition service for {audio_file}. File rejected.")
        os.remove(temp_wav_file)  # Remove the temporary WAV file
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        os.remove(temp_wav_file)

if __name__ == "__main__":
    audio_file = "/content/voice.mp3"  # Replace with your audio file path
    output_dir = "/content/output"  # Replace with your desired output directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    process_audio(audio_file, output_dir)
