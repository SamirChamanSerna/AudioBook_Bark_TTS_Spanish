from transformers import BarkModel, AutoProcessor
import torch
import nltk
import numpy as np
import scipy
import optimum


# import accelerate

# Set default tensor type to CUDA
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(torch.rand(10).device)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load in full precision
model = BarkModel.from_pretrained("suno/bark")
model = model.to(device)

# convert to bettertransformer
model = model.to_bettertransformer()

# enable CPU offload
# model.enable_cpu_offload()

# Generating speech
processor = AutoProcessor.from_pretrained("suno/bark")
sample_rate = model.generation_config.sample_rate

# Texto
script = """
Hey, have you heard about this new text-to-audio model called "Bark"? 
... 
be a game-changer in the world of text-to-audio technology.
""".replace(
    "\n", " "
).strip()

sentences = nltk.sent_tokenize(script)

# Voz a utilizar
voice_preset = "v2/en_speaker_6"
# quarter second of silence
silence = np.zeros(int(0.25 * sample_rate))

# GEN_TEMP = 0.6

# pieces = []
# for sentence in sentences:
#    audio_array = model.generate(
#        sentence,
#        history_prompt=voice_preset,
#        # num_beams=4,
#        temperature=1,
#        semantic_temperature=0.85,
#        do_sample=True,
#        early_stopping=False,
#    )
#    pieces += [audio_array, silence.copy()]

pieces = []
for sentence in sentences:
    audio_array = model.generate(sentence, history_prompt=voice_preset)
    pieces += [audio_array, silence.copy()]

audio_array = audio_array.cpu().numpy().squeeze()

# save them as a .wav file
scipy.io.wavfile.write(
    "Test1.wav", np.concatenate(pieces), rate=sample_rate, data=audio_array
)
