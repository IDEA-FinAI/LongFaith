#  ---------------------------------------------------------------------------------------------------------------------
PREDICT_COT_PROMPT = """You are provided with documents and a complex logical reasoning question.
You must refer to the documents to perform step-by-step logical reasoning and reach the correct answer.
Each reasoning step must be on a separate line, ending with a newline character.
The final answer must begin with `The answer is`, which is as much concise as possible without anyother words.

DOCUMENTS:
{docs}

QUESTION:
{question}"""

#  ---------------------------------------------------------------------------------------------------------------------

PREDICT_COC_PROMPT = """You are provided with documents and a complex logical reasoning question.
You must refer to the documents to perform step-by-step logical reasoning and reach the correct answer.
Each reasoning step must be on a separate line, ending with a newline character.
Cite the document properly during reasoning, e.g., `[1]`, `[2]`, etc.
The final answer must begin with `The answer is`, which is as much concise as possible without anyother words.

DOCUMENTS:
{docs}

QUESTION:
{question}"""

#  ---------------------------------------------------------------------------------------------------------------------

SYN_COT_PROMPT = """You are provided with documents, a complex logical reasoning question, and the correct answer.
You must refer to the documents to perform step-by-step logical reasoning and reach the correct answer.
Each reasoning step must be on a separate line, ending with a newline character.
End your reasoning with `The answer is` followed by the correct answer.

DOCUMENTS:
{docs}

QUESTION:
{question}

ANSWER:
{answer}"""
#  ---------------------------------------------------------------------------------------------------------------------

SYN_COC_PROMPT = """You are provided with documents, a complex logical reasoning question, and the correct answer.
You must refer to the documents to perform step-by-step logical reasoning and reach the correct answer.
Each reasoning step must be on a separate line, ending with a newline character.
Cite the document properly during reasoning, e.g., `[1]`, `[2]`, etc.
End your reasoning with `The answer is` followed by the correct answer.

DOCUMENTS:
{docs}

QUESTION:
{question}

ANSWER:
{answer}"""

#  ---------------------------------------------------------------------------------------------------------------------

SYN_AO_COT_PROMPT = """You are provided with a complex logical reasoning question and the correct answer.
You must perform step-by-step logical reasoning and reach the correct answer.
Each reasoning step must be on a separate line, ending with a newline character.
End your reasoning with `The answer is` followed by the correct answer.

QUESTION:
{question}

ANSWER:
{answer}"""


import re
import json

class DataManager:
    def __init__(self, dataset: str, mode: str, datasets: str):
        train_set_path = "train.jsonl"
        test_set_path = "test.jsonl"
        
        dataset_dir = f"{datasets}/{dataset}"
        
        self.datasets = datasets
        self.dataset = dataset
        self.train_set_path = f"{dataset_dir}/{train_set_path}"
        self.test_set_path = f"{dataset_dir}/{test_set_path}"
        
        if mode in ["train", "test"]:
            file_path = getattr(self, f"{mode}_set_path")
            data_set = []
            
            with open(file_path, "r") as f:
                for line in f:
                    sample = json.loads(line.strip())
                    data_set.append(sample)
                    
            setattr(self, f"{mode}_set", data_set)

    # Include all the documents in the context
    def build_pred_cot_prompt(self, sample):
        if self.datasets == "multihop":
            contexts = sample['contexts']
            context_text = "\n".join([f"{item['title']}: {item['paragraph_text']}" for item in contexts])
            question = sample["question_text"]
        else:
            context_text = sample['context']
            question = sample["input"]
        return PREDICT_COT_PROMPT.format(
            docs=context_text,
            question=question
        )

    # Include all the documents in the context with citations
    def build_pred_coc_prompt(self, sample):
        if self.datasets == "multihop":
            contexts = sample['contexts']
            context_text = "\n".join([f"[{i+1}] {item['title']}: {item['paragraph_text']}" for i, item in enumerate(contexts)])
            question = sample["question_text"]
        else:
            contexts = sample['context']
            if self.dataset in ["2wikimqa", "hotpotqa", "musique"]:
                context_text = re.sub(r'Passage (\d+):\n', r'[\1] ', contexts)
            elif self.dataset in ["qasper", "multifieldqa_en"]:
                context_text = '\n'.join([f'[{i+1}] {contexts[i*len(contexts)//20:(i+1)*len(contexts)//20]}' for i in range(20)])
            question = sample["input"]
        return PREDICT_COC_PROMPT.format(
            docs=context_text,
            question=question
        )

    # Only include the supporting facts in the context
    def build_syn_cot_prompt(self, sample):
        contexts = sample['contexts']
        context_text = "\n".join([f"{item['title']}: {item['paragraph_text']}" for item in contexts if item['is_supporting']])
        question = sample["question_text"]
        answer = sample["answers_objects"][0]["spans"][0]
        return SYN_COT_PROMPT.format(
            docs=context_text,
            question=question,
            answer=answer
        )

    # Only include the supporting facts in the context with citation
    def build_syn_coc_prompt(self, sample):
        contexts = sample['contexts']
        context_text = "\n".join([f"[{i+1}] {item['title']}: {item['paragraph_text']}" for i, item in enumerate(contexts) if item['is_supporting']])
        question = sample["question_text"]
        answer = sample["answers_objects"][0]["spans"][0]
        return SYN_COC_PROMPT.format(
            docs=context_text,
            question=question,
            answer=answer
        )
    
    # Only include the question and answer
    def build_syn_ao_cot_prompt(self, sample):
        question = sample["question_text"]
        answer = sample["answers_objects"][0]["spans"][0]
        return SYN_AO_COT_PROMPT.format(
            question=question,  
            answer=answer
        )
        