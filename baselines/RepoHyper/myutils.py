from transformers import T5ForConditionalGeneration, AutoTokenizer
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.models.unixcoder import UniXcoder

checkpoint = "Salesforce/codet5p-220m"
device = "cuda:3" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=512)
# model = T5ForConditionalGeneration.from_pretrained(checkpoint)
# model = model.encoder.to(device)

model = UniXcoder("microsoft/unixcoder-base").to(device)
