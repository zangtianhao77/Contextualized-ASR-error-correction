from huggingsound import SpeechRecognitionModel

# (1) jonatasgrosman/wav2vec2-large-xlsr-53-english model
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
audio_paths = ["mixed_speech.wav"]

transcriptions = model.transcribe(audio_paths)
print(type(transcriptions[0]['transcription']))

# (2) facebook model
model = SpeechRecognitionModel("facebook/wav2vec2-base-960h")
# audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
audio_paths = ["mixed_speech.wav"]

transcriptions = model.transcribe(audio_paths)
print(type(transcriptions[0]['transcription']))

# (3) facebook model
model = SpeechRecognitionModel("facebook/wav2vec2-large-960h")
# audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
audio_paths = ["mixed_speech.wav"]

transcriptions = model.transcribe(audio_paths)
print(type(transcriptions[0]['transcription']))

