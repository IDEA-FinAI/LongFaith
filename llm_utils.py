import os
import torch
from openai import OpenAI

def gpt_chat(model_name, user_prompt):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.gptsapi.net/v1"
    )
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.7,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return completion.choices[0].message.content

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_vllm(model_name):
    model = LLM(model=model_name, gpu_memory_utilization=0.90, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True, max_model_len=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def vllm_chat(llm, tokenizer, user_prompts, do_sample):
    if do_sample:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
    else:
        sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=1024)

    texts = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    outputs = llm.generate(texts, sampling_params)
    return outputs