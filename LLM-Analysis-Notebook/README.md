# Language Model Exploration and Analysis (GPT-2)

## Overview
This project focuses on exploring and analyzing the behavior of a pretrained Language Model using HuggingFace Transformers. The goal was to understand how a generative model responds to different types of prompts and evaluate its performance across multiple dimensions such as coherence, diversity, and context handling.

Instead of training a model from scratch, this project emphasizes practical usage of LLMs, which aligns with real-world AI engineering workflows.

---

## Model Used
- Model: `distilgpt2`
- Library: HuggingFace Transformers
- Reason for selection:
  - Lightweight and efficient
  - Suitable for experimentation in limited compute environments
  - Maintains strong text generation capability

---

## Project Structure
```

Advanced_LLM_Project/
├── LLM_Analysis_GPT2.ipynb        # Notebook with full workflow and explanations
├── LLM_Analysis_GPT2.py           # Script version for execution
├── outputs/
│   ├── plot_token_diversity.png
│   ├── plot_text_length.png
│   ├── plot_correlation_heatmap.png
└── README.md

````

---

## Key Components

### 1. Text Generation Experiments
The model was tested on different prompt types:
- Basic prompts (e.g., "Artificial Intelligence is")
- Creative writing prompts
- Factual questions
- Incomplete sentences
- Context-based prompts

This helped evaluate how prompt structure affects output.

---

### 2. Behavioral Analysis
The generated outputs were analyzed based on:
- Coherence of text
- Repetition patterns
- Context understanding
- Variation across prompt styles

---

### 3. Quantitative Analysis

#### Token Diversity Analysis
- Compared total tokens vs unique tokens
- Measured lexical diversity of generated text

#### Text Length Analysis
- Token count per prompt
- Word count per prompt

#### Correlation Analysis
- Correlation between:
  - Token count
  - Word count
  - Character count
  - Unique tokens
  - Lexical diversity

---

## Visual Outputs

### Token Diversity
![Token Diversity](outputs/plot_token_diversity.png)

### Text Length Analysis
![Text Length](outputs/plot_text_length.png)

### Correlation Heatmap
![Correlation Heatmap](outputs/plot_correlation_heatmap.png)

---

## Research Questions

1. How well does the model maintain context across different prompts?
2. How does prompt structure influence the generated output?
3. Does the model generate consistent and coherent responses?

---

## Key Insights

- Prompt wording has a strong impact on output quality and relevance  
- The model performs well on structured prompts but struggles with factual accuracy  
- Repetition can occur in longer generations  
- High lexical diversity indicates variation, but not necessarily correctness  
- Strong correlation observed between token, word, and character counts  

---

## Limitations

- No real-time knowledge (trained on past data)
- Can generate incorrect or fabricated information
- Limited deep contextual reasoning
- Sensitive to prompt phrasing

---

## Applications

- Content generation
- Chat-based systems
- Writing assistance tools
- Prompt engineering experiments

---

## Conclusion

This project demonstrates how pretrained language models can be used, tested, and analyzed in a structured way. It highlights both strengths and limitations of generative models and provides a foundation for building real-world AI applications on top of LLMs.

---

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch
- Matplotlib
- Seaborn

---

## How to Run

### Option 1: Notebook
Open `lm_analysis.ipynb` in Jupyter or Google Colab and run all cells.

### Option 2: Script
```bash
python lm_analysis.py
````

---

## Future Improvements

* Compare multiple LLMs (GPT-2 vs BERT vs newer models)
* Add evaluation metrics like BLEU or ROUGE
* Build a simple UI for interaction
* Integrate API-based models (OpenAI, etc.)

---