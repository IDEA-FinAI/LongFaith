# LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data 🚀

![LongFaith](https://github.com/IDEA-FinAI/LongFaith/blob/main/figures/longfaith_main.png)

## 📄 Paper & Resources
[![arXiv](https://img.shields.io/badge/Arxiv-2502.12583-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2502.12583)
[![hf_model_data](https://img.shields.io/badge/%F0%9F%A4%97-Models&Datasets-48A9DC)](https://huggingface.co/collections/cehao/longfaith-67b61f7b17ccb022c68ba22d)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Data-34A853)](https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🛠️ Setup

### Download Datasets
You can download the training and evaluation datasets from the following [Google Drive folder](https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC).

## 📂 File Directory

```
LongFaith_datasets/
│── longbench/
│   ├── 2wikimqa/
│   ├── hotpotqa/
│   ├── multifiedqa_en/
│   ├── musique/
│   ├── qasper/
│── multihop/
│   ├── 2wikimultihopqa/
│   ├── hotpotqa/
│   ├── musique/
│── longfaith_syn/
│   ├── gpt-4o/
│   ├── gpt-4o-mini/
│   ├── Meta-Llama-3.1-8B-Instruct/
│   ├── Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
│   ├── Qwen2.5-7B-Instruct/
```

### Create Environment
```bash
conda create -n longfaith python=3.11
conda activate longfaith
pip install -r requirements.txt
```

## 🚀 Model Running

```python
import transformers
import torch

model_id = "cehao/Meta-Llama-3.1-8B-Instruct-LongFaith-PO"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

## 📊 Evaluation

```bash
python predict.py
```

## 🙏 Acknowledgments
- Special thanks to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📓 Cite our Work
```python
@misc{yang2025longfaith,
      title={LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data}, 
      author={Cehao Yang and Xueyuan Lin and Chengjin Xu and Xuhui Jiang and Shengjie Ma and Aofan Liu and Hui Xiong and Jian Guo},
      year={2025},
      eprint={2502.12583},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12583}, 
}
```