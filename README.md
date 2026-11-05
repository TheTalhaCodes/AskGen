# 🧠 AskGen: Neural Question Generation (Seq2Seq + Attention)

> Ever read a Wikipedia page and thought, *"I wish someone would just test me on this?"* 
> 
> **AskGen** is a deep learning model built from scratch that does exactly that. Give it a piece of text and a specific answer, and it will craft the question that leads to that answer.

This project was developed as part of the Natural Language Processing (NLP) course. Unlike many modern AI projects that rely on massive pre-trained models (like GPT-4 or T5), **AskGen was built completely from scratch** to truly understand the mechanics of sequence-to-sequence learning.

---

## 🎯 The Challenge
*   **Input:** A context paragraph + a specific answer extracted from that context.
*   **Output:** A natural language question corresponding to the answer.

**Example:**
*   *Context:* "Paris is the capital and largest city of France..."
*   *Answer:* "Paris"
*   *Generated Question:* "What is the capital of France?"

---

## 🏗️ Architecture: The "No Magic, Just Math" Approach
We didn't just call an API. We built the brain piece by piece using **PyTorch**:

*   **The Encoder:** A Bidirectional LSTM that reads the context and the answer from both left-to-right and right-to-left to fully grasp the semantics.
*   **The Attention Mechanism (Bahdanau):** The secret sauce. At each generation step, it tells the Decoder *"Hey, focus on this specific word in the context!"* instead of forcing it to remember a 500-word paragraph all at once.
*   **The Decoder:** A unidirectional LSTM that generates the question word by word, using a **Teacher Forcing** strategy (ratio 0.5) during training to stay on track.

## 📦 Tech Stack
*   **Deep Learning:** PyTorch
*   **NLP & Tokenization:** spaCy (`en_core_web_sm`)
*   **Dataset:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset) via Hugging Face.
*   **Visualization:** Matplotlib (for the learning curves).

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up your environment:
```bash
git clone https://github.com/your-username/AskGen.git
cd AskGen
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
