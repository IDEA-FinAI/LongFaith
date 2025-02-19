import os
import re
import json
import argparse
from tqdm import tqdm
from datetime import datetime

from llm_utils import load_vllm, vllm_chat, gpt_chat
from data_manager import DataManager

def main(dataset, model_path, prompt, size, gtguide, ansonly):
    model_name = model_path.split("/")[-1] if "gpt" not in model_path else model_path
    # model_family = model_name.split("-")[0]
    model_family = model_name
    if "gpt" not in model_path:
        model, tokenizer = load_vllm(model_path)
    
    os.makedirs(f"logs_syn/{dataset}/{model_family}", exist_ok=True)
    log_file = os.path.join(f"logs_syn/{dataset}/{model_family}", f"{prompt}_{size}_{gtguide}_{ansonly}_{datetime.now().strftime('%m%d%H%M')}.txt")
    with open(log_file, 'w', encoding="utf-8") as log:
        log.write(f"Model loaded from {model_path}\n")
        data_manager = DataManager(dataset=dataset, mode="train", datasets="multihop")

        if size == "1k":
            proportion = {"2hop": 0, "3hop": 512, "4hop": 512}
        elif size == "2k":
            proportion = {"2hop": 512, "3hop": 512, "4hop": 1024}
        elif size == "4k":
            proportion = {"2hop": 1024, "3hop": 2048, "4hop": 1024}
        elif size == "8k":
            proportion = {"2hop": 3072, "3hop": 4096, "4hop": 1024}
        count = {"2hop": 0, "3hop": 0, "4hop": 0}
        filtered_samples = []
        for sample in data_manager.train_set:
            prefix = sample["question_id"][:4]
            if prefix == "2hop" or prefix == "3hop" or prefix == "4hop":
                if count[prefix] < proportion[prefix]:
                    filtered_samples.append(sample)
                    count[prefix] += 1
        
        syn_prompts = []
        for sample in filtered_samples:
            if ansonly == "contextual":
                if prompt == "cot":
                    if gtguide == "gtguide":
                        syn_prompt = data_manager.build_syn_cot_prompt(sample)
                    elif gtguide == "noguide":
                        syn_prompt = data_manager.build_pred_cot_prompt(sample)
                elif prompt == "coc":
                    if gtguide == "gtguide":
                        syn_prompt = data_manager.build_syn_coc_prompt(sample)
                    elif gtguide == "noguide":
                        syn_prompt = data_manager.build_pred_coc_prompt(sample)
            elif ansonly == "ansonly":
                syn_prompt = data_manager.build_syn_ao_cot_prompt(sample)
            syn_prompts.append(syn_prompt)
        
        if "gpt" not in model_path:
            outputs = vllm_chat(model, tokenizer, syn_prompts, do_sample=True)

            syn_sft = []
            for idx, (sample, output) in enumerate(tqdm(zip(filtered_samples, outputs), total=len(filtered_samples))):
                log.write(f"Sample {idx + 1}\n")
                log.write(f"Syn Prompt: {syn_prompts[idx]}\n")
                
                syn_solution = output.outputs[0].text
                log.write(f"Syn Solution: {syn_solution}\n")

                if prompt == "coc":
                    predict_prompt = data_manager.build_pred_coc_prompt(sample)
                elif prompt == "cot":
                    predict_prompt = data_manager.build_pred_cot_prompt(sample)
                
                syn_sft.append({
                    "instruction": predict_prompt,
                    "input": "",
                    "output": syn_solution
                })
        else:
            syn_sft = []
            for idx, sample in enumerate(tqdm(filtered_samples)):
                log.write(f"Sample {idx + 1}\n")
                log.write(f"Syn Prompt: {syn_prompts[idx]}\n")
                
                syn_solution = gpt_chat(model_path, syn_prompts[idx])
                log.write(f"Syn Solution: {syn_solution}\n")

                if prompt == "coc":
                    predict_prompt = data_manager.build_pred_coc_prompt(sample) 
                elif prompt == "cot":
                    predict_prompt = data_manager.build_pred_cot_prompt(sample)
                
                syn_sft.append({
                    "instruction": predict_prompt,
                    "input": "",
                    "output": syn_solution
                })
        
        os.makedirs(f"syn_sft/{dataset}/{model_family}", exist_ok=True)
        save_file = os.path.join(f"syn_sft/{dataset}/{model_family}", f"{prompt}_{size}_{gtguide}_{ansonly}.json")
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(syn_sft, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpot", "2wiki", "musique"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")  # meta-llama/Meta-Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct, gpt-4o-mini
    parser.add_argument("--prompt", type=str, default="coc", choices=["cot", "coc"])
    parser.add_argument("--size", type=str, default="2k", choices=["1k", "2k", "4k", "8k"])
    parser.add_argument("--gtguide", type=str, default="gtguide", choices=["gtguide", "noguide"])
    parser.add_argument("--ansonly", type=str, default="contextual", choices=["ansonly", "contextual"])
    args = parser.parse_args()
    main(args.dataset, args.model_path, args.prompt, args.size, args.gtguide, args.ansonly)