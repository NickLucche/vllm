# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Basic smoke test for google/translategemma-12b-it.

TranslateGemma is a Gemma3-based text translation model that uses the same
Gemma3ForConditionalGeneration architecture. This script validates that vLLM
can load the model and produce translations via its native chat template.

Usage:
    chg run -- python tests/test_translategemma.py
"""

from transformers import AutoProcessor

from vllm import LLM, SamplingParams

MODEL_NAME = "google/translategemma-12b-it"


def build_translation_prompt(
    processor: AutoProcessor,
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Build a translation prompt using the model's chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def main():
    print(f"Loading processor for {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"Initializing vLLM with {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        max_num_seqs=4,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    test_cases = [
        {
            "text": "Hello, how are you? I hope you are having a great day.",
            "source_lang": "en",
            "target_lang": "fr",
            "label": "English -> French",
        },
        {
            "text": "La inteligencia artificial está transformando el mundo.",
            "source_lang": "es",
            "target_lang": "en",
            "label": "Spanish -> English",
        },
        {
            "text": "Machine learning models require large amounts of data.",
            "source_lang": "en",
            "target_lang": "de",
            "label": "English -> German",
        },
    ]

    prompts = []
    for tc in test_cases:
        prompt = build_translation_prompt(
            processor, tc["text"], tc["source_lang"], tc["target_lang"]
        )
        prompts.append(prompt)

    print(f"\nRunning {len(prompts)} translation(s)...\n")
    outputs = llm.generate(prompts, sampling_params)

    all_passed = True
    for tc, output in zip(test_cases, outputs):
        generated = output.outputs[0].text.strip()
        print(f"[{tc['label']}]")
        print(f"  Input:  {tc['text']}")
        print(f"  Output: {generated}")
        if not generated:
            print("  ** FAIL: empty output **")
            all_passed = False
        print()

    if all_passed:
        print("All translation tests produced non-empty output.")
    else:
        print("SOME TESTS FAILED (empty output).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
