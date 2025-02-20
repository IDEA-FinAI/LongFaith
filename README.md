# LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data 🚀

![LongFaith](https://github.com/IDEA-FinAI/LongFaith/blob/main/figures/longfaith_main.png)

## 📄 Paper & Resources
[![arXiv](https://img.shields.io/badge/Arxiv-2502.12583-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2502.12583)
[![hf_paper](https://img.shields.io/badge/%F0%9F%A4%97-Paper-FF6F61)](https://huggingface.co/collections/cehao/longfaith-67b61f7b17ccb022c68ba22d)
[![hf_model_data](https://img.shields.io/badge/%F0%9F%A4%97-Models&Datasets-48A9DC)](https://huggingface.co/collections/cehao/longfaith-67b61f7b17ccb022c68ba22d)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Data-34A853)](https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🛠️ Setup

### Create Environment
```bash
conda create -n longfaith python=3.11
conda activate longfaith
pip install -r requirements.txt
```

### Install Dependencies
```bash
pip install transformers torch
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

---

🌟 **Happy Coding!** 🌟
```
