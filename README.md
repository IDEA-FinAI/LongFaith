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
<!-- Evaluation Datasets -->
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
<!-- Training Datasets -->
│── longfaith_syn/
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

The official implementation of Meta-Llama-3.1-8B-Instruct-LongFaith-PO is trained on LongFaith-PO synthesized by GPT-4o-mini.

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
    {"role": "user", "content": "You are provided with documents and a complex logical reasoning question.\nYou must refer to the documents to perform step-by-step logical reasoning and reach the correct answer.\nEach reasoning step must be on a separate line, ending with a newline character.\nCite the document properly during reasoning, e.g., `[1]`, `[2]`, etc.\nThe final answer must begin with `The answer is`, which is as much concise as possible without anyother words.\n\nDOCUMENTS:\n[1] The Spy in Black: \"The Spy in Black\" was filmed at Denham Studios, with location shooting at Northchurch Common in Berkhamsted, Hertfordshire and in Orkney, Scotland. The film wrapped production on 24 December 1938 and was released in the U.K. on 12 August 1939 – 22 days before the country again went to war with Germany. Its American premiere was held in New York City on 5 October of that year, and it went into general release two days later.\n[2] Thomas Fanshawe, 2nd Viscount Fanshawe: Thomas Fanshawe, 2nd Viscount Fanshawe (1632–1674) of Ware Park, Hertfordshire was an Irish peer and Member of Parliament. He was born to Thomas Fanshawe, 1st Viscount Fanshawe by his second wife Elizabeth Cockayne, the daughter of Sir William Cockayne, who served as the Lord Mayor of London in 1619.\n[3] Birth control movement in the United States: Birth control practices were generally adopted earlier in Europe than in the United States. Knowlton's book was reprinted in 1877 in England by Charles Bradlaugh and Annie Besant, with the goal of challenging Britain's obscenity laws. They were arrested (and later acquitted) but the publicity of their trial contributed to the formation, in 1877, of the Malthusian League -- the world's first birth control advocacy group -- which sought to limit population growth to avoid Thomas Malthus's dire predictions of exponential population growth leading to worldwide poverty and famine. By 1930, similar societies had been established in nearly all European countries, and birth control began to find acceptance in most Western European countries, except Catholic Ireland, Spain, and France. As the birth control societies spread across Europe, so did birth control clinics. The first birth control clinic in the world was established in the Netherlands in 1882, run by the Netherlands' first female physician, Aletta Jacobs. The first birth control clinic in England was established in 1921 by Marie Stopes, in London.\n[4] Wareside: Wareside is a small village and civil parish in the East Hertfordshire District, in the county of Hertfordshire. The population of the civil parish as of the 2011 census is 735. It is approximately 3 miles away from the town of Ware (from where it probably took its name) and the larger town of Hertford, the county town of Hertfordshire. Nearby villages include Widford, Hunsdon, Babbs Green and Bakers End. Nearby hamlets include Cold Christmas and Helham Green. The B1004 linking Ware to Bishop's Stortford goes through the village and the main A10 road can be picked up at Thundridge. Fanhams Hall Road also links Wareside back to Ware. Ware railway station on the Hertford East Branch Line is located two and a half miles away.\n[5] M. Visvesvaraya: Mokshagundam Viswesvarayya was born on 15 September 1861 in Muddenahalli village (now located in Chikkaballapura District, but part of Kolar district at the time of his birth) in the princely state of Mysore (now Karnataka), India. His father, Mokshagundam Srinivasa Sastry, was a school teacher and a noted Sanskrit scholar, while his mother, Venkatalakshamma, was a homemaker. His parents were from Mokshagundam, a village of Prakasam district in Andhra Pradesh.\n[6] Dilley sextuplets: The Dilley sextuplets (born May 25, 1993) are the United States' first set of surviving sextuplets, born to Becki and Keith Dilley in Indianapolis, Indiana, United States. They are, in birth order;\n[7] Hertfordshire: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.\n[8] Margaret Sanger: Margaret Higgins Sanger (born Margaret Louise Higgins, September 14, 1879 -- September 6, 1966, also known as Margaret Sanger Slee) was an American birth control activist, sex educator, writer, and nurse. Sanger popularized the term ``birth control '', opened the first birth control clinic in the United States, and established organizations that evolved into the Planned Parenthood Federation of America.\n[9] Harry Potter (film series): Filming of the series began at Leavesden Studios, Hertfordshire, England, in September 2000 and ended in December 2010, with post-production on the final film lasting until summer 2011. Leavesden Studios was the main base for filming Harry Potter, and it opened to the public as a studio tour in 2012 (renamed as Warner Bros. Studios, Leavesden).\n[10] Edgar Anstey: Edgar Anstey (16 February 1907 in Watford, Hertfordshire, England – 26 September 1987 in London, England), was a leading British documentary film-maker.\n[11] East of England: The East of England is one of nine official regions of England at the first level of NUTS for statistical purposes. It was created in 1994 and was adopted for statistics from 1999. It includes the ceremonial counties of Bedfordshire, Cambridgeshire, Essex, Hertfordshire, Norfolk and Suffolk. Essex has the highest population in the region.\n[12] Thomas Plumer Halsey: Thomas Plumer Halsey MP (26 January 1815 – 24 April 1854) was a Member of Parliament for Hertfordshire from 1846 to 1854.\n[13] Watford Rural: Watford Rural is a civil parish in the Three Rivers District of Hertfordshire, England. Located approximately northwest of central London and adjacent to the Greater London boundary, it is an urbanised parish characterised by suburban residential development. The local council is Watford Rural Parish Council. The parish covers South Oxhey and Carpenders Park, which although part of the Watford urban area, are outside the borough of Watford. The parish was created in 1894 when the ancient Watford parish was split into urban and rural parishes. At the 2001 census it had a population of 20,250.\n[14] Ralph Trustees Limited: Ralph Trustees Limited is a family run private hotel group based in England with a portfolio of four hotels operating in the four and five star sector. Their hotels include The Grove (Hertfordshire), The Athenaeum (London), The Runnymede (Surrey) and 23 Greengarden House (London).\n[15] Cyril Dumpleton: Cyril Walter Dumpleton (25 June 1897 – 1 October 1966) was a British Labour Party politician who served as the Member of Parliament (MP) for the St Albans division of Hertfordshire from 1945 to 1950.\n[16] Bengeo Rural: Bengeo Rural is a civil parish in the East Hertfordshire district of Hertfordshire, England. According to the 2001 census it had a population of 601, increasing at the 2011 Census to 644. The parish includes the villages of Tonwell and Chapmore End.\n[17] Demographics of the European Union: The most populous member state is Germany, with an estimated 82.8 million people, and the least populous member state is Malta with 0.4 million. Birth rates in the EU are low with the average woman having 1.6 children. The highest birth - rates are found in Ireland with 16.876 births per thousand people per year and France with 13.013 births per thousand people per year. Germany has the lowest birth rate in Europe with 8.221 births per thousand people per year.\n[18] Barkway: Barkway is a long-established village and civil parish in the North Hertfordshire district of Hertfordshire, England, about five miles south-east of Royston, 35 miles from London and 15 miles from the centre of Cambridge. The Prime Meridian passes a mile or so to the west of Barkway.\n[19] Untitled (The Birth): Untitled (The Birth) is a 1938 tempera painting by American artist Jacob Lawrence, located in the Indianapolis Museum of Art, which is in Indianapolis, Indiana. Depicting a scene of childbirth in flat, geometric forms and bright colors, it is very much a product of the Harlem Renaissance.\n[20] Wyddial: Wyddial is a village and civil parish in the East Hertfordshire district of Hertfordshire, England. It is located around a mile and a half north-east of Buntingford (OS grid reference ), and lies due north of Greenwich on the Prime Meridian.\n\nQUESTION:\nWhat year saw the creation of the region where the county of Hertfordshire is located?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

## 📊 Evaluation

```bash
python predict.py --model_path cehao/Meta-Llama-3.1-8B-Instruct-LongFaith-PO --datasets multihop --prompt coc
python predict.py --model_path cehao/Meta-Llama-3.1-8B-Instruct-LongFaith-PO --datasets longbench --prompt coc
```

## 🎯 Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning and preference optimization, which provides an efficient training pipiline. The hyperparameters are given in our paper.

## 🙏 Acknowledgments
Special thanks to:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)

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

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.