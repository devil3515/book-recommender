# Semantic Book Recommender with LLMs

## Overview
This project is a **semantic book recommender system** powered by **Large Language Models (LLMs), embeddings, and vector databases**. It allows users to receive book recommendations based on textual descriptions, categories, and emotional tones using **Hugging Face embeddings** and **ChromaDB**.

## Project Structure
- `data-exploration.ipynb` - Exploratory Data Analysis (EDA) on book datasets.
- `sentiment-analysis.ipynb` - Analyzes book descriptions for sentiment.
- `text_classification.ipynb` - Classifies books into predefined categories.
- `vector_search.ipynb` - Implements vector-based search using embeddings.
- `gradio-dashboard.py` - **Gradio-powered UI** for interactive book recommendations.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/devil3515/book-recommender.git
cd book-recommender
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Gradio App
```bash
python gradio-dashboard.py
```
This will start a web-based dashboard for **semantic book recommendations**.

## Features
✅ **Semantic book search** using **LLM-powered embeddings**.  
✅ **Vector-based retrieval** for personalized book recommendations.  
✅ **Emotion-aware filtering** (Happy, Surprising, Sad, etc.).  
✅ **Interactive Gradio UI** for easy book discovery.  

## Technologies Used
- **Python**
- **Pandas, NumPy** for data processing
- **Hugging Face Transformers** for embeddings
- **ChromaDB** for vector search
- **Gradio** for UI

## License
This project is licensed under the **MIT License**.
