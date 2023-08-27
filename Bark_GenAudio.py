from transformers import BarkModel, AutoProcessor
import torch
import scipy
import optimum
import accelerate
import os

# os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# model = BarkModel.from_pretrained("suno/bark")

device = "cuda" if torch.cuda.is_available() else "cpu"

# load in fp16
model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
# model = BetterTransformer.transform(model, keep_original_model=False)

# enable CPU offload
# model.enable_cpu_offload()

# Generating speech
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/es_speaker_0"

inputs = processor("Hola, buenos días", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# save them as a .wav file
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out2.wav", rate=sample_rate, data=audio_array)
