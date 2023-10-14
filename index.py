from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "hello, world"

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform

audio = output.squeeze().numpy()
sf.write("output_audio.wav", audio, 22050)
