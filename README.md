# LLM-Based Intelligent Process Planning (CAPP) System

## Overview
An AI-powered Computer-Aided Process Planning system that uses
Large Language Models to automatically generate, validate, and
optimize manufacturing process routes.

## Tech Stack
- Python 3.11
- PyTorch + HuggingFace Transformers
- NSGA-II Multi-objective Optimization (DEAP)
- FSM-based Validation
- Pandas | NumPy | Matplotlib | Jupyter

## Project Structure

## Week 1 — Completed
- [x] Feature Tokenizer (word to token encoding)
- [x] Synthetic Dataset Generator (100 manufacturing parts)
- [x] Jupyter Notebook exploration & visualization

## Week 2 — Completed
- [x] FSM Validator (operation sequence validation)
- [x] Data Preprocessor (label encoding + one-hot encoding)
- [x] Random Forest Classifier (75% accuracy)
- [x] Jupyter analysis notebook with visualizations

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/feature_tokenizer.py
python src/data_generator.py
python src/fsm_validator.py
python src/preprocessor.py
python src/ml_model.py
```

## Author
Radhana123 | Data Science Portfolio Project