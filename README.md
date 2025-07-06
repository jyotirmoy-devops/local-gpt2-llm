# Word-Level GPT-2 Trainer (Beginner-Friendly)

This project is a simple, educational implementation of a word-level GPT-2 style text generator using TensorFlow and Keras. It is designed for beginners who want to understand how text generation models work and how to train them on their own data.

## Features
- **Word-level text generation** for coherent sentences
- Clean, step-by-step code with detailed comments
- Easy to customize with your own text
- Interactive text generation in the terminal
- Training progress visualization
- Uses NLTK for word tokenization
- **API deployment with FastAPI** for easy integration

---

## Table of Contents
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Customizing the Model](#customizing-the-model)
- [Tips for Better Results](#tips-for-better-results)
- [Deployment (FastAPI)](#deployment-fastapi)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [License](#license)

---

## How It Works
1. **Loads and cleans sample text** (or your own text)
2. **Tokenizes text into words** using NLTK
3. **Builds a word-level vocabulary** (maps each word to a number)
4. **Creates training sequences** (input and target pairs with 10-word context by default)
5. **Defines a word-level GPT-2 style model** using LSTM layers
6. **Trains the model** to predict the next word in a sequence
7. **Generates new text** based on a starting phrase
8. **Provides an interactive mode** for you to try your own prompts

---

## Setup & Installation
1. **Clone this repository** (or copy `word_level_gpt2.py` to your project folder).
2. **Install the required Python packages**:
   ```bash
   pip install tensorflow numpy matplotlib nltk fastapi uvicorn
   ```
3. **Download NLTK data** (the script will do this automatically, but you can also run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
4. **(Optional) Check your Python version**  
   Python 3.7+ is recommended.

---

## Usage
1. **Run the script**:
   ```bash
   python word_level_gpt2.py
   ```
2. **What happens next?**
   - The script will print progress for each step: loading data, preprocessing, building the model, training, and generating text.
   - After training, it will show sample generated texts with different starting phrases.
   - You can use the interactive mode to generate your own text prompts.
3. **Interactive Text Generation**
   After training, you can use the interactive function:
   ```python
   >>> from word_level_gpt2 import interactive_generation_words
   >>> interactive_generation_words()
   ```
   Type a starting phrase and see the model generate text!

---

## Customizing the Model
- **Use your own text:**  
  Replace the `sample_text` variable in `word_level_gpt2.py` with your own text, or load from a file:
  ```python
  with open('your_file.txt', 'r') as f:
      sample_text = f.read()
  ```
- **Change model size:**  
  Adjust these parameters in the code:
  - `embedding_dim = 128` (embedding dimensions)
  - `rnn_units = 256` (LSTM units)
  - `batch_size = 16` (training batch size)
- **Increase sequence length:**  
  Change `seq_length` (default is 10) to 20 or 30 for more context.
- **Train longer:**  
  Increase the `epochs` parameter in `model.fit()` for better results.
- **Experiment with temperature:**  
  The `temperature` parameter in `generate_text_words()` controls randomness:
  - `0.5` = More conservative, predictable text
  - `1.0` = More creative, surprising text
- **Expand vocabulary:**  
  Increase `vocab_size` to allow more unique words.

---

## Tips for Better Results
- Use **more training text** (the more, the better!)
- **Increase sequence length** for more context
- **Train for more epochs** (30+)
- **Experiment with temperature** for different creativity levels
- **Try different starting phrases**
- **Use interactive generation** to explore the model's creativity

---

## Deployment (FastAPI)

You can deploy your trained model as a web API using FastAPI. This allows you (or others) to generate text from any application or device.

### **1. Install FastAPI and Uvicorn**
```bash
pip install fastapi uvicorn
```

### **2. Run the FastAPI Server**
```bash
uvicorn app:app --reload
```
- The server will start at `http://127.0.0.1:8000/`

### **3. Use the `/generate` Endpoint**
- **POST** to `/generate` with a JSON payload:
  - `prompt`: The starting text
  - `num_generate`: (optional) Number of words to generate (default: 20)
  - `temperature`: (optional) Sampling temperature (default: 0.7)

#### **Example with `curl`**
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "the quick brown fox", "num_generate": 15, "temperature": 0.7}'
```

#### **Example with Python**
```python
import requests
response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "the quick brown fox", "num_generate": 15, "temperature": 0.7}
)
print(response.json()["generated_text"])
```

### **4. Interactive API Docs**
- Visit `http://127.0.0.1:8000/docs` in your browser for an interactive Swagger UI.

---

## Troubleshooting
- **Import errors?**  
  Make sure you have installed all required packages: `tensorflow`, `numpy`, `matplotlib`, `nltk`, `fastapi`, `uvicorn`.
- **NLTK errors?**  
  Make sure you have downloaded the required NLTK data (`punkt`, `stopwords`).
- **Out of memory?**  
  Reduce `embedding_dim`, `rnn_units`, or `batch_size`.
- **Weird output?**  
  Use more training data, train longer, or try different temperature values.
- **Training too slow?**  
  Reduce `rnn_units` or `batch_size` for faster training.

---

## Technical Details
- **Model**: 2-layer LSTM with dropout
- **Embedding Dimension**: 128
- **LSTM Units**: 256 per layer
- **Vocabulary Size**: 500+ (configurable)
- **Sequence Length**: 10 words (configurable)
- **Training Examples**: Based on your text size
- **Batch Size**: 16 (configurable)
- **Epochs**: 30 (configurable)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Text Generation**: Temperature-controlled sampling

---

## Changelog

### Version 3.0.0 (Current)
**Date**: July 6, 2025

#### 🚀 Major Changes
- **Switched to word-level GPT-2 model** for much more coherent, real-sentence generation.
- **Created `word_level_gpt2.py`**: All training, generation, and interactive code is now word-based.
- **Removed old character-level code** (`main.py`) to avoid confusion and focus on best practices.
- **Added FastAPI deployment**: New `app.py` exposes a `/generate` endpoint for text generation via web API.
- **Updated README.md**: Now fully beginner-friendly, with clear instructions for training, customization, and deployment.
- **Added Deployment section**: Step-by-step guide for running the FastAPI server and using the API.
- **Improved troubleshooting and tips**: More guidance for common issues and best practices.

#### 🛠️ Technical Improvements
- Uses NLTK for robust word tokenization.
- Model and vocabulary are now word-based for better context and output.
- Interactive generation and API both use the same model and vocabulary.
- All code and documentation now focus on word-level best practices.

---

## License
This project is for educational purposes. Feel free to use, modify, and share!

---

**Happy experimenting!**  
If you have questions or want to learn more, check out the comments in `word_level_gpt2.py`—they explain each step in detail.

**Next Steps:**
- Try training on your own text data
- Experiment with different model parameters
- Use interactive generation to explore creativity
- Add model saving/loading functionality if needed
- Deploy as an API for easy integration 