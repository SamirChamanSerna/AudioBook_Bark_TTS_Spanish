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
# Necesita optimum
# model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)

# convert to bettertransformer
# Al parecer esto hace que el modelo completo alucine más.
# can gain 20% to 30% in speed with zero performance degradation.
model = model.to_bettertransformer()

# enable CPU offload
# 80% reduction in memory footprint
# model.enable_cpu_offload()

# Generating speech
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/es_speaker_4"

inputs = processor(
    "Hola, esto es algo más avanzado.",
    voice_preset=voice_preset,
)
# Basic
# audio_array = model.generate(**inputs)

# Advanced
audio_array = model.generate(
    **inputs,
    # num_beams=4,
    temperature=1,
    semantic_temperature=0.85,
    do_sample=True,
    early_stopping=False
)

audio_array = audio_array.cpu().numpy().squeeze()

# save them as a .wav file
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out_Test16.wav", rate=sample_rate, data=audio_array)
