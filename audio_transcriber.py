"""
A script to transcribe audio files from MP3 to text using the OpenAI Whisper model.

This script is designed to be run from the command line and takes a folder
path as an argument. It finds all .mp3 files in that folder, transcribes them,
and saves the transcription to a .txt file with the same name.

Key improvements over the original version:
- Uses command-line arguments for flexibility (no hardcoded paths).
- Converts MP3 to WAV in-memory to avoid creating temporary files.
- Automatically uses a GPU (CUDA) if available for faster processing.
- Includes error handling to gracefully skip files that can't be processed.
- Better structured with a main function and __name__ == "__main__" guard.
- More detailed comments and type hinting for clarity.
"""

import os
import io
import argparse
from pathlib import Path
import torch
import torchaudio
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def setup_model_and_processor():
    """
    Loads the Whisper model and processor, and moves the model to the appropriate device (GPU if available).
    """
    print("Loading Whisper model and processor...")
    # Determine the device to run the model on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        model.eval()
        return model, processor, device
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return None, None, None

def transcribe_audio(model: WhisperForConditionalGeneration, processor: WhisperProcessor, device: str, mp3_path: Path):
    """
    Transcribes a single MP3 file and saves the transcription to a text file.

    Args:
        model: The pre-loaded Whisper model.
        processor: The pre-loaded Whisper processor.
        device: The device to run inference on ('cuda' or 'cpu').
        mp3_path (Path): The path to the input MP3 file.
    """
    print(f"\nProcessing file: {mp3_path.name}")

    if not mp3_path.exists():
        print(f"Error: File not found at {mp3_path}")
        return

    try:
        # 1. Convert MP3 to WAV in-memory
        audio = AudioSegment.from_mp3(mp3_path)
        # Export to a buffer instead of a file
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # 2. Load and resample audio using torchaudio
        speech_array, sampling_rate = torchaudio.load(buffer)
        
        # Resample if necessary to match Whisper's required 16kHz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)

        # 3. Prepare features for the model
        input_features = processor(
            speech_array.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)

        # 4. Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # 5. Save transcription to a .txt file
        txt_path = mp3_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcription.strip())
        
        print(f"✅ Transcription saved to: {txt_path.name}")

    except Exception as e:
        print(f"❌ Failed to process {mp3_path.name}. Error: {e}")


def main():
    """
    Main function to parse arguments and process all MP3 files in a directory.
    """
    parser = argparse.ArgumentParser(description="Transcribe all MP3 files in a given folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing MP3 files.")
    args = parser.parse_args()

    folder = Path(args.folder_path)
    if not folder.is_dir():
        print(f"Error: Provided path '{folder}' is not a valid directory.")
        return

    model, processor, device = setup_model_and_processor()
    if not model:
        return # Exit if model setup failed

    # Find all .mp3 files in the specified folder
    mp3_files = list(folder.glob("*.mp3"))
    if not mp3_files:
        print(f"No .mp3 files found in {folder}")
        return
        
    print(f"\nFound {len(mp3_files)} MP3 file(s) to transcribe.")
    
    for file_path in mp3_files:
        transcribe_audio(model, processor, device, file_path)

if __name__ == "__main__":
    main()
