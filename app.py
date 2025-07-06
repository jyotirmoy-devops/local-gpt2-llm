from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import os
import sys

# Import model and vocabulary from word_level_gpt2.py
from word_level_gpt2 import (
    build_word_model, word_to_idx, idx_to_word, seq_length, embedding_dim, rnn_units, vocab_size, generate_text_words
)

# Load the trained model weights (assume model is trained in the same session)
# If you want to load from a file, you can use model.load_weights('path_to_weights.h5')

app = FastAPI(title="Word-Level GPT-2 Text Generation API")

class GenerationRequest(BaseModel):
    prompt: str
    num_generate: int = 20
    temperature: float = 0.7

@app.post("/generate")
def generate(request: GenerationRequest):
    try:
        # Build a fresh generation model (batch_size=1)
        model = build_word_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        model.build(input_shape=(1, None))
        # Optionally, load weights from a file here if needed
        # model.load_weights('word_level_gpt2_weights.h5')
        # For now, we assume weights are in memory (from training session)
        # Generate text
        generated = generate_text_words(
            model,
            start_string=request.prompt,
            num_generate=request.num_generate,
            temperature=request.temperature
        )
        return {"generated_text": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Word-Level GPT-2 Text Generation API. Use POST /generate with a prompt."} 