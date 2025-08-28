import os

os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

hf_hub_cache_dir = os.path.join('.', 'data', 'hf_cache')
os.makedirs(hf_hub_cache_dir, exist_ok=True)
os.environ['HF_HUB_CACHE'] = hf_hub_cache_dir

# # Demo 01
# import gradio as gr
# from transformers import pipeline

# gr.Interface.from_pipeline(pipeline("text-classification", model="Johnson8187/Chinese-Emotion")).launch(share=True)


# Demo 02
import gradio as gr
from transformers import pipeline

gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch(share=True)

