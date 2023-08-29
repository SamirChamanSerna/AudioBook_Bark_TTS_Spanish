from transformers import BarkModel, AutoProcessor
import torch
import scipy
import optimum
import accelerate

# Set default tensor type to CUDA
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.rand(10).device)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load in full precision
model = BarkModel.from_pretrained("suno/bark")
model = model.to(device)

# load in fp16
# No parece que funciona pero el archivo de audio no contiene nada
# model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
model = model.to_bettertransformer()

# enable CPU offload
model.enable_cpu_offload()

# Generating speech
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/es_speaker_0"

inputs = processor(
    "Hola. Esta es una prueba con unas cuantas optimizaciones",
    voice_preset=voice_preset,
)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# save them as a .wav file
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out3.wav", rate=sample_rate, data=audio_array)
