import re
import os
import json
import time
import argparse
from tqdm import tqdm
from datetime import datetime

from llm_utils import load_vllm, vllm_chat
from metrics import compute_exact, compute_f1, compute_subem, compute_attr_metrics
from data_manager import DataManager

def main(model_path, prompt, datasets):
    model_name = model_path.split("/")[-1]

    if datasets == "multihop":
        sub_datasets = ["hotpotqa", "2wikimultihopqa", "musique"]
    elif datasets == "longbench":
        sub_datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]

    model, tokenizer = load_vllm(model_path)
    for dataset in sub_datasets:
        os.makedirs(f"logs_{datasets}/{dataset}", exist_ok=True)
        log_file = os.path.join(f"logs_{datasets}/{dataset}", f"{model_name}_{prompt}_{datetime.now().strftime('%m%d%H%M')}.txt")

        with open(log_file, 'w') as log: 
            log.write(f"Model loaded from {model_path}\n")
            data_manager = DataManager(datasets=datasets, dataset=dataset, mode="test")

            total_qa_em_score = 0
            total_qa_f1_score = 0
            total_qa_subem_score = 0
            total_attr_precision = 0
            total_attr_recall = 0
            total_attr_f1 = 0
            total_len = 0
            max_len = 0
            
            if datasets == "multihop":
                if dataset == "hotpotqa":
                    type_metrics = {}
                else:
                    hop_metrics = {}

            predict_prompts = []

            start_time = time.time()
            
            for sample in data_manager.test_set:
                if prompt == "cot":
                    predict_prompt = data_manager.build_pred_cot_prompt(sample)
                elif prompt == "coc":
                    predict_prompt = data_manager.build_pred_coc_prompt(sample)
                predict_prompts.append(predict_prompt)

            outputs = vllm_chat(model, tokenizer, predict_prompts, do_sample=False)

            for idx, (sample, output) in enumerate(tqdm(zip(data_manager.test_set, outputs), total=len(data_manager.test_set))):
                log.write(f"Sample {idx + 1}\n")
                if datasets == "multihop":
                    a_golds = sample["answers_objects"][0]["spans"]
                else:
                    a_golds = sample["answers"]
                
                log.write(f"QA prompt: {predict_prompts[idx]}\n")
                log.write(f"Ground truth answer: {a_golds}\n")
                if datasets == "multihop":
                    support_facts = []
                    for context in sample["contexts"]:
                        if context["is_supporting"]:
                            support_facts.append(context["idx"] + 1)
                    log.write(f"Supporting facts: {support_facts}\n")

                    if dataset == "hotpotqa":
                        type = sample["type"]
                    else:
                        hop_num = len(support_facts)
                
                response = output.outputs[0].text
                log.write(f"LLM QA Response: \n{response}\n")
                
                qa_em_score = 0
                qa_f1_score = 0            
                match = re.search(r'the answer is (.*)', response.lower())
                if match is None:
                    log.write("No answer found in the response\n")
                else:
                    a_pred = match.group(1)
                    for a_gold in a_golds:
                        qa_em_score = max(qa_em_score, compute_exact(a_gold, a_pred))
                        qa_f1_score = max(qa_f1_score, compute_f1(a_gold, a_pred))
                        
                qa_subem_score = 0
                for a_gold in a_golds:
                    qa_subem_score = max(qa_subem_score, compute_subem(a_gold, response))

                if datasets == "multihop":
                    attr_precision, attr_recall, attr_f1 = compute_attr_metrics(response, support_facts)

                log.write(f"QA EM score: {qa_em_score}\n")
                log.write(f"QA F1 score: {qa_f1_score:.3f}\n")
                log.write(f"QA SubEM score: {qa_subem_score}\n")
                if datasets == "multihop":
                    log.write(f"Attribution precision: {attr_precision:.3f}\n")
                    log.write(f"Attribution recall: {attr_recall:.3f}\n")
                    log.write(f"Attribution F1: {attr_f1:.3f}\n")

                total_qa_em_score += qa_em_score
                total_qa_f1_score += qa_f1_score
                total_qa_subem_score += qa_subem_score
                total_len += len(predict_prompts[idx])
                max_len = max(max_len, len(predict_prompts[idx]))

                if datasets == "multihop":
                    total_attr_precision += attr_precision
                    total_attr_recall += attr_recall
                    total_attr_f1 += attr_f1
                
                    if dataset == "hotpotqa":
                        if type not in type_metrics:
                            type_metrics[type] = {
                                "qa_em": 0,
                                "qa_f1": 0,
                                "qa_subem": 0,
                                "attr_precision": 0,
                                "attr_recall": 0,
                                "attr_f1": 0,
                                "count": 0,
                                "len": 0,
                                "max_len": 0
                            }

                        type_metrics[type]["qa_em"] += qa_em_score
                        type_metrics[type]["qa_f1"] += qa_f1_score
                        type_metrics[type]["qa_subem"] += qa_subem_score
                        type_metrics[type]["attr_precision"] += attr_precision
                        type_metrics[type]["attr_recall"] += attr_recall
                        type_metrics[type]["attr_f1"] += attr_f1
                        type_metrics[type]["count"] += 1
                        type_metrics[type]["len"] += len(predict_prompts[idx])
                        type_metrics[type]["max_len"] = max(type_metrics[type]["max_len"], len(predict_prompts[idx]))
                    else:
                        if hop_num not in hop_metrics:
                            hop_metrics[hop_num] = {
                                "qa_em": 0,
                                "qa_f1": 0,
                                "qa_subem": 0,
                                "attr_precision": 0,
                                "attr_recall": 0,
                                "attr_f1": 0,
                                "count": 0,
                                "len": 0,
                                "max_len": 0
                            }

                        hop_metrics[hop_num]["qa_em"] += qa_em_score
                        hop_metrics[hop_num]["qa_f1"] += qa_f1_score
                        hop_metrics[hop_num]["qa_subem"] += qa_subem_score
                        hop_metrics[hop_num]["attr_precision"] += attr_precision
                        hop_metrics[hop_num]["attr_recall"] += attr_recall
                        hop_metrics[hop_num]["attr_f1"] += attr_f1
                        hop_metrics[hop_num]["count"] += 1
                        hop_metrics[hop_num]["len"] += len(predict_prompts[idx])
                        hop_metrics[hop_num]["max_len"] = max(hop_metrics[hop_num]["max_len"], len(predict_prompts[idx]))

                if (idx + 1) % 50 == 0:
                    log.write("*" * 50 + "\n")
                    log.write(f"Prompt Length: {total_len / (idx + 1):.3f}\n")
                    log.write(f"Max Prompt Length: {max_len}\n")
                    log.write(f"Total QA EM: {total_qa_em_score / (idx + 1):.3f}\n")
                    log.write(f"Total QA F1: {total_qa_f1_score / (idx + 1):.3f}\n")
                    log.write(f"Total QA SubEM: {total_qa_subem_score / (idx + 1):.3f}\n")
                    if datasets == "multihop":
                        log.write(f"Total Attribution Precision: {total_attr_precision / (idx + 1):.3f}\n")
                        log.write(f"Total Attribution Recall: {total_attr_recall / (idx + 1):.3f}\n")
                        log.write(f"Total Attribution F1: {total_attr_f1 / (idx + 1):.3f}\n")

                log.write("*" * 50 + "\n")

            if datasets == "multihop":
                if dataset == "hotpotqa":
                    for type in type_metrics:
                        log.write(f"Type: {type} Count: {type_metrics[type]['count']}\n")
                        log.write(f"Prompt Length: {type_metrics[type]['len'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"Max Prompt Length: {type_metrics[type]['max_len']}\n")
                        log.write(f"QA EM: {type_metrics[type]['qa_em'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"QA F1: {type_metrics[type]['qa_f1'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"QA SubEM: {type_metrics[type]['qa_subem'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"Attribution Precision: {type_metrics[type]['attr_precision'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"Attribution Recall: {type_metrics[type]['attr_recall'] / type_metrics[type]['count']:.3f}\n")
                        log.write(f"Attribution F1: {type_metrics[type]['attr_f1'] / type_metrics[type]['count']:.3f}\n")
                        log.write("*" * 50 + "\n")
                else:
                    for hop_num in hop_metrics:
                        log.write(f"Hop number: {hop_num} Count: {hop_metrics[hop_num]['count']}\n")
                        log.write(f"Prompt Length: {hop_metrics[hop_num]['len'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"Max Prompt Length: {hop_metrics[hop_num]['max_len']}\n")
                        log.write(f"QA EM: {hop_metrics[hop_num]['qa_em'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"QA F1: {hop_metrics[hop_num]['qa_f1'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"QA SubEM: {hop_metrics[hop_num]['qa_subem'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"Attribution Precision: {hop_metrics[hop_num]['attr_precision'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"Attribution Recall: {hop_metrics[hop_num]['attr_recall'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write(f"Attribution F1: {hop_metrics[hop_num]['attr_f1'] / hop_metrics[hop_num]['count']:.3f}\n")
                        log.write("*" * 50 + "\n")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            log.write(f"Total time cost: {elapsed_time:.2f}s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")  # Qwen/Qwen2.5-7B-Instruct LLM-Research/Meta-Llama-3.1-70B-Instruct
    parser.add_argument("--prompt", type=str, default="coc", choices=["cot", "coc"])
    parser.add_argument("--datasets", type=str, default="multihop", choices=["multihop", "longbench"])
    args = parser.parse_args()
    main(args.model_path, args.prompt, args.datasets)