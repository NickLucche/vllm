# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Basic offline inference for gemma3n
import vllm

# model_name = "facebook/mbart-large-en-ro"
model_name = "facebook/bart-large-cnn"

# NOTE important to NOT split the item, set it in config (IF NOT ALREADY SET)
llm = vllm.LLM(
    model=model_name,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=1024,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.5,
    dtype="float16",
    disable_chunked_mm_input=True,
)

#   hf_overrides={"architectures": ["MBartForConditionalGeneration"]}


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
print(
    "PROMPT encoder:",
    tokenizer.encode("The president of the United States is", add_special_tokens=False),
    "\n",
)
print("PROMPT decoder:", tokenizer.encode("Donald", add_special_tokens=False), "\n")

params = vllm.SamplingParams(temperature=0.0, max_tokens=20)
# FIXME working with explicit encoder/decoder prompt only!
outputs = llm.generate(
    [
        # {
        #     "prompt": "The president of the United States is",
        # },
        {  # Test explicit encoder/decoder prompt
        # <s> for empty prompt
            "encoder_prompt": {
                "prompt": "The president of the United States is",
            },
            "decoder_prompt": "<s>Donald",
        },
        { 
        # <s> for empty prompt
            "encoder_prompt": {
                "prompt": "<s>",
            },
            "decoder_prompt": "<s>Ronald McDonald is",
        },
    ],
    sampling_params=params,
)
# TODO output is really sensible to the BOS token which should always be present in decoder promtp!
# No <s> token:
# output: . The president of the United States is president of the United States is president of the United States is
# With <s> token:
# output:  Trump is president of the United States. The president of the United States is president of the United States

# NOTE same thing for empty encoder_promt: should be set to BOS. So ""->"<s>"
for o in outputs:
    generated_text = o.outputs[0].text
    print("output:", generated_text)
