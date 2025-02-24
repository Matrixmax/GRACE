---
configs:
- config_name: default
  data_files:
  - split: cross_file_first
    path: data/cross_file_first-*
  - split: cross_file_random
    path: data/cross_file_random-*
  - split: in_file
    path: data/in_file-*
dataset_info:
  features:
  - name: repo_name
    dtype: string
  - name: file_path
    dtype: string
  - name: context
    list:
    - name: identifier
      dtype: string
    - name: path
      dtype: string
    - name: snippet
      dtype: string
  - name: import_statement
    dtype: string
  - name: token_num
    dtype: int64
  - name: cropped_code
    dtype: string
  - name: all_code
    dtype: string
  - name: next_line
    dtype: string
  - name: gold_snippet_index
    dtype: int64
  - name: created_at
    dtype: string
  - name: level
    dtype: string
  splits:
  - name: cross_file_first
    num_bytes: 504528431
    num_examples: 8033
  - name: cross_file_random
    num_bytes: 467242455
    num_examples: 7618
  - name: in_file
    num_bytes: 488999100
    num_examples: 7910
  download_size: 472994299
  dataset_size: 1460769986
license: cc
task_categories:
- text-generation
language:
- en
tags:
- code
---
# RepoBench v1.1 (Java)

## Introduction

This dataset presents the **Java** portion of [RepoBench](https://arxiv.org/abs/2306.03091) v1.1 (ICLR 2024). The data encompasses a collection from GitHub, spanning the period from **October 6th to December 31st, 2023**. With a commitment to data integrity, we've implemented a deduplication process based on file content against the Stack v2 dataset (coming soon), aiming to mitigate data leakage and memorization concerns.

## Resources and Links

- [Paper](https://arxiv.org/abs/2306.03091)
- [GitHub](https://github.com/Leolty/repobench)
- [Dataset Introduction](https://github.com/Leolty/repobench/blob/main/data/README.md)

## FAQs

- **Q:** What do the features in the dataset mean?
  
  **A:** Imagine you're coding and you want to write the next line of your code. The dataset provides you the following information:
    - `repo_name` (string): the name of the repository
    - `file_path` (string): the path of the current file
    - `context` (list): the cross-file code snippets that might be helpful for writing the next line:
      - `identifier` (string): the identifier of the code snippet
      - `path` (string): the path of the code snippet
      - `snippet` (string): the code snippet
    - `import_statement` (string): the import statement of the current file
    - `cropped_code` (string): the cropped code of the current file (up to previous 120 lines)
    - `all_code` (string): the entire code of the current file (not cropped)
    - `next_line` (string): the next line of the code (this serves as the target)
    - `gold_snippet_index` (int): the index of the gold snippet in the context (which will be used in next line, just for reference, you should not use this for next line prediction)
    - `created_at` (string): the creation time of the repository
    - `level` (string): the level of next line completion, which is measured by the number of tokens for the whole prompt (including all the context, import statement, cropped code and some neccessary separator tokens)

- **Q:** How does the level be defined?

  **A:** The level is determined by the number of tokens for the whole prompt (including all the context, import statement, cropped code and some neccessary separator tokens). The token number is calculated by the tokenizer of GPT-4 by using [tiktoken](https://github.com/openai/tiktoken). The following table shows the level definition:

    | Level | Prompt Length (Number of Tokens) |
    |-------|------------------------|
    | 2k    | 640 - 1,600            |
    | 4k    | 1,600 - 3,600          |
    | 8k    | 3,600 - 7,200          |
    | 12k   | 7,200 - 10,800         |
    | 16k   | 10,800 - 14,400        |
    | 24k   | 14,400 - 21,600        |
    | 32k   | 21,600 - 28,800        |
    | 64k   | 28,800 - 57,600        |
    | 128k  | 57,600 - 100,000       |

- **Q:** What does the different splits mean?

  **A:** The dataset is split into three parts:
    - `cross_file_first`: the next line of code utilizes content from a cross-file code snippet and it is its first usage within current file.
    - `cross_file_random`: the next line of code utilizes content from a cross-file code snippet and it is NOT its first usage within current file.
    - `in_file`: the next line of code does not utilize content from a cross-file code snippet.

- **Q:** How to construct the prompt for next line prediction?

  **A:** We hereby provide the official implementation for constructing prompts. Please note that the methods described below are not necessarily the optimal way of construction. Reordering, retrieval argumentation, or employing different cropping/construction techniques could potentially lead to varying degrees of improvement. Ensure that your model evaluations are conducted in a fair manner.

    ```python
    import re

    def construct_prompt(
        data: dict, 
        language: str = "java",
        tokenizer= None,
        max_token_nums: int = 15800
        ) -> str:
        """
        Construct the prompt for next line prediction.

        :param data: data point from the dataset
        :param language: the language of the code
        :param tokenizer: the tokenizer of the evaluation model
        :param max_token_nums: the maximum number of tokens constraint for the prompt

        :return: the constructed prompt
        """

        # comment symbol for different languages
        comment_symbol = "#" if language == "python" else "//"

        # construct the cross-file prompt and in-file prompt separately
        # cross-file prompt
        cross_file_prompt = f"{comment_symbol} Repo Name: {data['repo_name']}\n"

        for snippet in data['context']:
            cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{snippet['snippet']}" + "\n\n"
        
        # in-file prompt
        in_file_prompt = f"{comment_symbol} Path: {data['file_path']}\n{data['import_statement']}\n{data['cropped_code']}\n"

        # if we assign the tokenizer and the max_token_nums, we will truncate the cross-file prompt to meet the constraint
        if tokenizer is not None and max_token_nums is not None:
            
            cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
            in_file_prompt_token_nums = len(tokenizer.encode(in_file_prompt))

            exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - max_token_nums

            if exceed_token_nums > 0:
                # split the cross-file prompt into lines
                cross_file_prompt_lines = cross_file_prompt.split("\n")
                # drop lines from end until the extra token number is less than 0
                for i in range(len(repo_prompt_lines)-1, -1, -1):
                    extra_token_num -= len(tokenizer.encode(cross_file_prompt_lines[i]))
                    if extra_token_num < 0:
                        break
                
                # join the lines back
                cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"
        
        # combine the cross-file prompt and in-file prompt
        prompt = cross_file_prompt + in_file_prompt

        # normalize some empty lines
        prompt = re.sub(r'\n{4,}', '\n\n', prompt)

        return prompt
    ```

- **Q:** How to load the dataset?

  **A:** You can simply use the following code to load the dataset:

    ```python
    from datasets import load_dataset

    dataset = load_dataset("tianyang/repobench_java_v1.1")
    ```

    To construct the prompt for next line prediction, you can refer to the official implementation provided in the previous question and use the `construct_prompt` function to construct the prompt, for example:

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

    prompt = construct_prompt(dataset['cross_file_first'][0], language="java", tokenizer=tokenizer, max_token_nums=15800)
    ```

- **Q:** How often will the dataset be updated?

  **A:** We plan to update the dataset every three months, but there might be slight delays considering the time required for data crawling and our own schedules. If you require updated data, please feel free to contact us, and we can coordinate the timing and expedite the process.

- **Q:** What models should I use to evaluate the dataset?

  **A:** RepoBench is designed to evaluate base models, not those that have been instruction fine-tuned. Please use base models for evaluation.

- **Q:** I am training a new model but the knowledge cutoff date is after the dataset's. Can you provide me with the latest data?

  **A:** Sure! We are happy to provide you with the latest data (even customized data with specific requirements). Please feel free to contact us.

- **Q:** Can I opt-out?
    
  **A:** Yes, you can opt-out your repository from the dataset. Please check [Am I in RepoBench?](https://huggingface.co/spaces/tianyang/in-the-repobench), we will upload the raw data of the repository information we crawled at least 15 days before the dataset creation and release. We will respect your decision and remove your repository from the dataset if you opt-out.

## Citation

If you find RepoBench useful in your research, please consider citing the paper using the following BibTeX entry:

```bibtex
@misc{liu2023repobench,
      title={RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems}, 
      author={Tianyang Liu and Canwen Xu and Julian McAuley},
      year={2024},
      url={https://arxiv.org/abs/2306.03091},
      booktitle={International Conference on Learning Representations}
}
```

Your interest and contributions to RepoBench are immensely valued. Happy coding! 🚀