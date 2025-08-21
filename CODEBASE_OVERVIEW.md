# Codebase Overview: Multilingual Question Answering System

## Core Components

### 1. Main Application Files

#### `app.py`
- **Purpose**: Web-based interface for the QA system using Streamlit
- **Key Features**:
  - Interactive Q&A interface
  - Model comparison dashboard
  - Performance metrics visualization
  - Sample questions for testing
- **Dependencies**: Streamlit, model_handler.py, language_detector.py, evaluation.py

#### `cli.py`
- **Purpose**: Command-line interface for the QA system
- **Key Features**:
  - Interactive Q&A mode
  - Batch processing of questions from files
  - Model comparison functionality
  - Support for multiple languages (English and Arabic)
- **Dependencies**: model_handler.py, language_detector.py, evaluation.py

### 2. Core Modules

#### `models/model_handler.py`
- **Purpose**: Manages loading and inference of various QA models
- **Key Features**:
  - Handles multiple model architectures (BERT, DistilBERT, Multilingual BERT)
  - Caches models for performance
  - Supports both English and Arabic models
  - Handles device management (CPU/GPU)

### 3. Utility Modules (`utils/`)

#### `data_loader.py`
- **Purpose**: Handles loading and preprocessing of QA datasets
- **Key Features**:
  - Supports multiple dataset formats
  - Handles data splitting and preprocessing
  - Manages data caching

#### `evaluation.py`
- **Purpose**: Implements evaluation metrics for QA models
- **Key Features**:
  - Exact match and F1 score calculation
  - Performance comparison between models
  - Detailed metrics reporting

#### `language_detector.py`
- **Purpose**: Detects language of input text
- **Key Features**:
  - Identifies English and Arabic text
  - Handles mixed-language content
  - Used for automatic model selection

### 4. Jupyter Notebooks

#### `arabic_qa_training.ipynb`
- **Purpose**: Training and evaluation of Arabic QA models
- **Contents**:
  - Data loading and preprocessing for Arabic
  - Model training code
  - Evaluation metrics and visualization

#### `english_qa_training.ipynb`
- **Purpose**: Training and evaluation of English QA models
- **Contents**:
  - Data loading and preprocessing for English
  - Model training code
  - Evaluation metrics and visualization

#### `model_comparison.ipynb`
- **Purpose**: Compare performance of different QA models
- **Contents**:
  - Side-by-side model evaluation
  - Performance metrics comparison
  - Visualization of results

## Usage Guide

### Running the Application

1. **Web Interface** (Recommended for most users):
   ```bash
   streamlit run app.py
   ```

2. **Command Line Interface**:
   ```bash
   python cli.py
   ```

- Example Context & Question 
   ```bash
   python cli.py --context  "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest." --question "How much area does the Amazon basin cover?"
   ```

   - Example Context & Question 
   ```bash
    python cli.py --context  "الذكاء الاصطناعي (AI) هو قدرة الأنظمة الحاسوبية على أداء مهام ترتبط عادةً بالذكاء البشري، مثل التعلم، والتفكير، وحل المشكلات، والإدراك، واتخاذ القرارات. وهو مجال بحثي في علوم الحاسوب يُطوّر ويدرس أساليب وبرامج تُمكّن الآلات من إدراك بيئتها واستخدام التعلم والذكاء لاتخاذ إجراءات تُعزز فرصها في تحقيق الأهداف المحددة." --question "ما هو الذكاء الاصطناعي؟"
   ```

3. **Jupyter Notebooks** (For development/training):
   - Open in Jupyter Lab/Notebook
   - Run cells sequentially

### File Dependencies

- **Essential Files** (required for running the application):
  - `app.py` or `cli.py` (choose one interface)
  - `models/model_handler.py`
  - `utils/*.py` (all utility modules)

- **Training/Development Files** (optional for end users):
  - `*_training.ipynb` notebooks
  - `model_comparison.ipynb`

## Recommendations
1. For end users: Use either `app.py` (web) or `cli.py` (command line)
2. For developers: The Jupyter notebooks provide training and evaluation code
3. The utility modules (`utils/`) are required dependencies but don't need to be run directly

## Planned vs Implemented Models

Planned in notebooks (`english_qa_training.ipynb`, `arabic_qa_training.ipynb`, `model_comparison.ipynb`):

- English: `bert-large-uncased-whole-word-masking-finetuned-squad`, `distilbert-base-cased-distilled-squad`, `deepset/roberta-base-squad2`, `twmkn9/albert-base-v2-squad2`
- Multilingual/Arabic: `bert-base-multilingual-cased`, `deepset/xlm-roberta-base-squad2`, `aubmindlab/bert-base-arabertv2-finetuned-squadv1`, `salti/bert-base-multilingual-cased-finetuned-squad`

Implemented at runtime (app/CLI):

- BERT (English), DistilBERT (English), mBERT (Multilingual), `salti/bert-base-multilingual-cased-finetuned-squad`

Reasons for divergence:

- The runtime intentionally limits to a focused set for reliability; planned-only models remain in notebooks for experimentation

Tokenization/chunking differences:

- Training code uses `max_length=384` and `doc_stride=128` during feature prep
- Inference pipeline truncates to model max length (detailed path uses `max_length=512`) without sliding windows; keep contexts concise to avoid truncation
