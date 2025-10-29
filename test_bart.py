# Basic offline inference for gemma3n

import vllm
import librosa
from PIL import Image
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

# 
# model_name = "facebook/mbart-large-en-ro"
model_name = "facebook/bart-large-cnn"

# NOTE important to NOT split the item, set it in config (IF NOT ALREADY SET)
llm = vllm.LLM(model=model_name, tensor_parallel_size=1, enforce_eager=True, max_model_len=1024,
 max_num_seqs=1, max_num_batched_tokens=1024, gpu_memory_utilization=0.5, dtype="float32", disable_chunked_mm_input=True)

#   hf_overrides={"architectures": ["MBartForConditionalGeneration"]}

# image = Image.open("image.jpg")
# image2 = ImageAsset("cherry_blossom").pil_image
# image = ImageAsset("cherry_blossom").pil_image
# assert image.size != image2.size
# audio = librosa.load("/home/nicolo/speech.wav", sr=16_000)
# audio2 = AudioAsset("mary_had_lamb").audio_and_sample_rate

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
print("PROMPT encoder:", tokenizer.encode("The president of the United States is", add_special_tokens=False), "\n")
print("PROMPT decoder:", tokenizer.encode("Donald", add_special_tokens=False), "\n")

params = vllm.SamplingParams(temperature=0.2, max_tokens=64)
outputs = llm.generate(
    [
        # TODO should default to encoder prompt I think
        # {
        #     "prompt": "The president of the United States is",
        # },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "The president of the United States is",
        },
        "decoder_prompt": "Donald",
    },
    ],
    sampling_params=params,
)

for o in outputs:
    generated_text = o.outputs[0].text
    print("output:", generated_text)