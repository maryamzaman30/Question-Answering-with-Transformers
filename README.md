# Natural Language Processing Internship - Elevvo Pathways

This project is a part of my **Natural Language Processing Internship** at **Elevvo Pathways**, Egypt.

## Internship Details

- **Company:** Elevvo Pathways, Egypt
- **Internship Period:** July - August 2025

# Question Answering with Transformers - Multilingual Question Answering System

A comprehensive question answering system built with Streamlit that supports both English and Arabic languages using **real transformer models** from Hugging Face. This application provides an intuitive web interface for interactive question answering, model comparison, and performance evaluation using state-of-the-art NLP models.

> **Current Implementation**: The system uses **real transformer models** BERT, DistilBERT, and multilingual BERT variants for actual question answering. All models are downloaded from Hugging Face and perform real inference on your questions.

## 🌟 Features

- **Real Transformer Models**: Uses actual BERT, DistilBERT, and multilingual BERT models
- **Multilingual Support**: Works with both English and Arabic text using specialized models
- **Multiple AI Models**: Compare different transformer models side-by-side
- **Interactive Web Interface**: Easy-to-use Streamlit interface with multiple tabs
- **Language Auto-Detection**: Automatically detects if your question is in English or Arabic
- **Model Performance Comparison**: Compare different models with real metrics
- **Real Evaluation Metrics**: Get actual performance scores on SQuAD v1.1 dataset
- **Sample Questions**: Pre-loaded examples to test the system
- **Real-time Model Loading**: Models are downloaded and cached for fast subsequent use

- Link to App Online - 
- Link to Video Demo - https://youtu.be/T1uwcs-tU5E

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all necessary packages including:
   - `streamlit` - Web interface
   - `transformers` - Hugging Face transformer models
   - `torch` - PyTorch backend
   - `datasets` - SQuAD dataset loading
   - `pandas` - Data processing

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py --server.port 5000
```

The application will open in your browser at `http://localhost:5000`

**Note**: On first use, models will be downloaded from Hugging Face (may take 2-5 minutes). Subsequent uses will be much faster as models are cached locally.

## 📱 Application Interface

The app has four main tabs, each serving different purposes:

### 1. Question Answering Tab

**What it does**: Interactive question answering interface using real transformer models.

**How to use**:
1. Enter a context paragraph in the text area (the background information)
2. Type your question in the question field
3. Select which transformer model you want to use from the dropdown
4. Adjust the confidence threshold if needed
5. Click "Get Answer" to receive your response

**Features**:
- **Real BERT/DistilBERT models** for English questions
- **Real multilingual models** for cross-language support (mBERT and mBERT fine-tuned on SQuAD)
- Shows actual confidence scores from transformer models
- Displays answer position in the context
- Color-coded confidence levels (High/Medium/Low)
- Real-time language detection and model selection

**Available Models**:
- **BERT (English)**: `bert-large-uncased-whole-word-masking-finetuned-squad`
- **DistilBERT (English)**: `distilbert-base-uncased-distilled-squad`
- **Multilingual BERT**: `bert-base-multilingual-cased`
- **Multilingual BERT (fine-tuned)**: `salti/bert-base-multilingual-cased-finetuned-squad`

**Example Usage**:
- Context: "Paris is the capital city of France. It is located on the Seine River and has a population of over 2 million people."
- Question: "What is the capital of France?"
- Answer: "Paris" (with real confidence score from BERT)

### 2. Model Comparison Tab

**What it does**: Compare how different transformer models answer the same question with real performance metrics.

**How to use**:
1. Enter your context and question
2. Click "Compare Models" to evaluate all available models
3. View results showing each model's answer, confidence, and response time

**What you'll see**:
- Side-by-side comparison of all transformer models
- Real confidence scores from each model
- Actual processing time for each model
- Summary showing the best performing model
- Progress bar showing evaluation progress

**Why use this**: 
- Find which transformer model works best for your type of questions
- See how different models interpret the same question
- Choose the fastest or most accurate model for your needs
- Compare real performance across different model architectures

### 3. Evaluation Tab

**What it does**: Test model performance on the SQuAD v1.1 dataset with real evaluation metrics.

**How to use**:
1. Upload a custom dataset file (JSON format) or use the built-in SQuAD evaluation
2. Select which model to evaluate
3. Click "Run Evaluation" or "Evaluate on Sample SQuAD Data"
4. Review detailed performance metrics

**Metrics provided**:
- **Exact Match**: Percentage of answers that match exactly (real SQuAD metric)
- **F1 Score**: Overall performance score (0-1, higher is better)
- **Total Questions**: Number of questions processed
- **Sample Predictions**: Examples of model answers vs expected answers

**Built-in Evaluation**:
- **SQuAD v1.1 Validation**: 50 sample questions from the official validation set
- **Real Model Performance**: Actual transformer model results on benchmark data
- **Performance Comparison**: See how your selected model performs on standard QA tasks

**Use cases**:
- Test how well a model performs on the SQuAD benchmark
- Compare model accuracy on different types of questions
- Validate model performance before deploying
- Benchmark against industry standards

### 4. Sample Questions Tab

**What it does**: Pre-loaded examples to quickly test the system with real transformer models.

**Available samples**:

**English Examples**:
- Amazon rainforest area questions
- Artificial intelligence definitions
- General knowledge scenarios

**Arabic Examples**:
- Egyptian capital questions
- AI definitions in Arabic
- Cultural and geographical questions

**How to use**:
1. Browse through the sample questions
2. Click on any example to automatically fill the Question Answering tab
3. Modify the examples or use them as templates
4. Test with real transformer model inference

## 🔧 Advanced Features

### Language Detection
The system automatically detects whether your question is in English or Arabic and selects appropriate transformer models accordingly.

### Model Loading and Caching
- Models are downloaded from Hugging Face on first use
- Subsequent uses load from local cache for fast inference
- GPU acceleration is automatically detected and used when available
- Memory management with automatic cleanup

### Confidence Scoring
Each answer comes with a real confidence score from the transformer model:
- **High (70%+)**: Very reliable answer from the model
- **Medium (30-70%)**: Moderately reliable answer
- **Low (<30%)**: Less reliable answer

### Model Selection
Choose from a focused set of transformer models:
- **BERT**: Great for general English questions (high accuracy)
- **DistilBERT**: Faster, lighter version of BERT (good speed/accuracy trade-off)
- **Multilingual Models**: Work with both languages (including a fine-tuned mBERT for QA)

## 📊 Understanding the Results

### Answer Components
- **Answer Text**: The extracted answer from your context (real transformer output)
- **Confidence Score**: How certain the transformer model is (0-1 scale)
- **Start/End Position**: Where in the context the answer was found
- **Processing Time**: How long the model took to respond

### Performance Metrics
- **Exact Match**: Binary score - did the model get the exact right answer?
- **F1 Score**: Balanced measure of precision and recall
- **Confidence**: Model's own assessment of answer quality

## 🛠️ Technical Architecture

### Core Components
- **Model Handler**: Manages loading and running different Hugging Face transformer models
- **Language Detector**: Identifies English vs Arabic text using advanced NLP techniques
- **Evaluation Engine**: Calculates real performance metrics on SQuAD v1.1
- **Streamlit Interface**: Provides the web-based user interface

### Supported Models
The system supports HuggingFace transformer models including:
- **BERT variants** for English (fine-tuned on SQuAD)
- **Multilingual models** for cross-language support
- **DistilBERT** for faster processing

### Data Processing
- **SQuAD v1.1 Integration**: Direct loading from Hugging Face datasets
- **Real-time Evaluation**: Live model performance testing
- **Custom Dataset Support**: Upload your own QA datasets for evaluation

## 🎯 Use Cases

### Educational
- Students can ask questions about reading passages using real AI models
- Teachers can test comprehension with custom contexts
- Language learners can practice with both English and Arabic using specialized models

### Research
- Compare real transformer model performance on different domains
- Evaluate new models against SQuAD v1.1 benchmark
- Analyze confidence patterns across different model architectures

### Business
- Customer service Q&A automation with real AI
- Document analysis and information extraction
- Multilingual content understanding using specialized models

## 🔍 Tips for Best Results

### Writing Good Contexts
- Include relevant information that contains the answer
- Keep contexts focused and not too long (models work best with focused passages)
- Make sure the context is in the same language as your question

### Asking Good Questions
- Be specific and clear
- Ask questions that can be answered from the context

### Model Selection
- Use **BERT** for English questions (high accuracy)
- Use **multilingual models** when working with mixed languages or Arabic (fine-tuned mBERT recommended)
- Choose **DistilBERT** when you need faster responses

## 🚨 Troubleshooting

### Common Issues
- **Model loading slow**: First use downloads models (2-5 minutes), subsequent uses are fast
- **No answer found**: Make sure your context contains the information needed
- **Low confidence**: Try rephrasing your question or providing more context
- **Wrong language detected**: Manually select the appropriate model if needed

### Performance Tips
- Shorter contexts usually process faster
- Clear, direct questions get better results
- Make sure context and question are in the same language
- Use GPU if available for faster inference

## 📈 Future Enhancements

This system is designed to be extensible. Potential improvements include:
- Support for more languages and specialized models
- Integration with additional transformer architectures
- Advanced evaluation metrics and visualization
- Batch processing capabilities
- Custom model fine-tuning interface
- Real-time model performance monitoring

## 🔍 Planned vs Implemented Models (important)

Differences between notebooks (planned) and runtime (app/CLI):

- Implemented in runtime:
  - **BERT (English)**: `bert-large-uncased-whole-word-masking-finetuned-squad`
  - **DistilBERT (English)**: `distilbert-base-uncased-distilled-squad`
  - **Multilingual BERT**: `bert-base-multilingual-cased`, `salti/bert-base-multilingual-cased-finetuned-squad`

- Planned only (not in runtime; available in notebooks):
  - `deepset/roberta-base-squad2`
  - `twmkn9/albert-base-v2-squad2`
  - `deepset/xlm-roberta-base-squad2`
  - `aubmindlab/bert-base-arabertv2-finetuned-squadv1`

Notes on tokenization and context length:

- Training uses sliding windows (`max_length=384`, `doc_stride=128`).
- Inference uses truncation with max length (512 in the detailed path). Keep contexts focused.

The application and CLI include only the implemented models listed above.

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets used.

---

## ❓ Need Help?

Try the Sample Questions tab to see the system in action, or start with simple questions in the Question Answering tab! The system now uses real transformer models for actual AI-powered question answering, shorter and simpler the passage, better the answer with higher confidence.

## 📖 Documentation

For more detailed documentation about the codebase structure and components, please see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md).
