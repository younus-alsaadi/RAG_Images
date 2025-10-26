# RAG_Images

RAG_Images is an implementation of Retrieval-Augmented Generation (RAG) for image-rich documents.
It extends traditional RAG pipelines by integrating vision-language models (VLMs) to handle PDFs containing images, charts, and multimodal content.

Key Features

1- Qwen2.5-VL Integration – Performs high-quality OCR on both English and German images and PDFs.

2- VisRAG / ColPali Modules – Enable retrieval and reasoning over visual elements such as charts and figures.

3- Hybrid RAG Pipeline – Combines Qwen2.5-VL for OCR with VisRAG for visual document understanding, allowing efficient retrieval and generation across multimodal content.

Use Case
This system is especially useful when:

- You need to retrieve only relevant pages from large visual PDFs before applying OCR.

- You want to extract information from charts, diagrams, or images where text-based methods fail.
 
## Requirements

- Python 3.12 or later

### Install Python using MiniConda

1. Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2. Create a new environment:
   ```bash
   conda create -n rag_images python=3.12
3) Activate the environment:
    ```bash
    $ conda activate rag_images


### Then install all required packages using pip:

  ```bash
    $ pip install -r requirements.txt