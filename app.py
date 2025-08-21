import streamlit as st
import time
import os
import json

# Environment flags must be set before importing any Transformers-related code
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0: all, 1: INFO, 2: WARNING, 3: ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # Ensure Transformers does not try TensorFlow

# Import real implementations
from models.model_handler import ModelHandler
from utils.language_detector import LanguageDetector
from utils.evaluation import QAEvaluator

# Suppress TensorFlow info logs and oneDNN notices (already set above)


# Initialize session state with real implementations
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = ModelHandler()
if 'language_detector' not in st.session_state:
    st.session_state.language_detector = LanguageDetector()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = QAEvaluator()

st.set_page_config(
    page_title="Multilingual Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# Reduce Streamlit console noise
try:
    st.set_option('logger.level', 'error')
except Exception:
    pass

st.title("🤖 Multilingual Question Answering System")
st.markdown("**Support for English and Arabic languages using transformer models**")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection (runtime-supported models)
    model_options = {
        "BERT (English)": "bert-large-uncased-whole-word-masking-finetuned-squad",
        "DistilBERT (English)": "distilbert-base-uncased-distilled-squad",
        "Multilingual (mBERT)": "bert-base-multilingual-cased",
        "Arabic (mBERT QA)": "salti/bert-base-multilingual-cased-finetuned-squad"
    }
    
    selected_model = st.selectbox(
        "Select Model:",
        options=list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model]
    
    # Auto-detect language option
    auto_detect = st.checkbox("Auto-detect language", value=True)
    
    # Manual language selection (when auto-detect is off)
    if not auto_detect:
        language = st.selectbox(
            "Select Language:",
            options=["English", "Arabic"],
            index=0
        )
    
    # Basic configuration only
    
    # Model loading info
    st.subheader("ℹ️ Model Information")
    st.info("**Note:** Models are downloaded and loaded on first use. This may take a few minutes initially, but subsequent uses will be faster.")
    
    # Show loaded models status
    st.subheader("📊 Model Status")
    if hasattr(st.session_state.model_handler, 'models'):
        loaded_models = list(st.session_state.model_handler.models.keys())
        if loaded_models:
            st.success(f"✅ {len(loaded_models)} models loaded")
            for model in loaded_models[:3]:  # Show first 3
                st.write(f"• {model.split('/')[-1]}")
            if len(loaded_models) > 3:
                st.write(f"• ... and {len(loaded_models) - 3} more")
        else:
            st.info("No models loaded yet. They will be loaded on first use.")
    else:
        st.info("Model handler initialized. Models will be loaded on first use.")

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["💬 Question Answering", "📊 Model Comparison", "📈 Evaluation", "🧪 Sample Questions"])

with tab1:
    st.header("Ask Questions")
    
    # Input fields
    context = st.text_area(
        "Context/Passage:",
        height=200,
        placeholder="Enter the context or passage here...",
        value=st.session_state.get('sample_context', "")
    )
    
    question = st.text_input(
        "Question:",
        placeholder="What do you want to know about the context?",
        value=st.session_state.get('sample_question', "")
    )
    
    if st.button("Get Answer", type="primary", disabled=not (context and question)):
        try:
            # Detect language if auto-detect is enabled
            if auto_detect:
                detected_lang = st.session_state.language_detector.detect_language(question + " " + context)
                st.info(f"Detected language: {detected_lang}")
                
                # Select appropriate model based on detected language
                if detected_lang == "Arabic":
                    # Prefer a fine-tuned multilingual model for Arabic
                    model_name = "salti/bert-base-multilingual-cased-finetuned-squad"
                else:
                    model_name = model_options[selected_model]
            
            # Load model and get answer
            with st.spinner("Loading model and processing..."):
                answer_data = st.session_state.model_handler.get_answer(
                    context, question, model_name
                )
                # Arabic low-confidence fallback sequence
                if auto_detect and detected_lang == "Arabic" and answer_data.get('score', 0.0) < 0.05:
                    fallback_candidates = []
                    if model_name != "salti/bert-base-multilingual-cased-finetuned-squad":
                        fallback_candidates.append("salti/bert-base-multilingual-cased-finetuned-squad")
                    fallback_candidates.append("bert-base-multilingual-cased")
                    tried = set([model_name])
                    for fb in fallback_candidates:
                        if fb in tried:
                            continue
                        st.warning("Low confidence detected. Trying an Arabic fallback model...")
                        temp = st.session_state.model_handler.get_answer(context, question, fb)
                        tried.add(fb)
                        if temp.get('score', 0.0) >= answer_data.get('score', 0.0):
                            answer_data = temp
                            model_name = fb
                            break
                # Fallback sequence if the selected model fails
                if isinstance(answer_data.get('answer'), str) and answer_data['answer'].startswith("Error:"):
                    fallbacks = [
                        "distilbert-base-uncased-distilled-squad",
                        "bert-base-multilingual-cased",
                        "bert-large-uncased-whole-word-masking-finetuned-squad"
                    ]
                    for fb in fallbacks:
                        if fb == model_name:
                            continue
                        st.warning("Selected model failed to load. Trying a fallback model...")
                        model_name = fb
                        answer_data = st.session_state.model_handler.get_answer(
                            context, question, model_name
                        )
                        if not (isinstance(answer_data.get('answer'), str) and answer_data['answer'].startswith("Error:")):
                            break
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Answer:")
                    st.success(answer_data['answer'])
                    st.caption(f"Confidence: {answer_data['score']:.3f}")
                
                with col2:
                    st.subheader("Details:")
                    st.write(f"**Start Position:** {answer_data['start']}")
                    st.write(f"**End Position:** {answer_data['end']}")
                    st.write(f"**Model Used:** {model_name.split('/')[-1]}")
                    
                    # Confidence display
                    confidence_level = "High" if answer_data['score'] > 0.7 else "Medium" if answer_data['score'] > 0.3 else "Low"
                    st.markdown(f"**Confidence Level:** {confidence_level}")
                    
                    # Simple progress bar for confidence
                    st.progress(answer_data['score'])
                    st.caption(f"Score: {answer_data['score']:.3f}")
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

with tab2:
    st.header("Model Performance Comparison")
    
    if st.button("Run Model Comparison"):
        if context and question:
            with st.spinner("Comparing models..."):
                comparison_results = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_models = len(model_options)
                
                for idx, (model_display_name, model_id) in enumerate(model_options.items()):
                    try:
                        # Update progress
                        progress_bar.progress((idx + 1) / total_models, text=f"Evaluating {model_display_name}...")
                        
                        start_time = time.time()
                        result = st.session_state.model_handler.get_answer(
                            context, question, model_id
                        )
                        used_display_name = model_display_name
                        if isinstance(result.get('answer'), str) and result['answer'].startswith("Error:"):
                            for fb in ["distilbert-base-uncased-distilled-squad", "bert-base-multilingual-cased", "bert-large-uncased-whole-word-masking-finetuned-squad"]:
                                if model_id == fb:
                                    continue
                                temp = st.session_state.model_handler.get_answer(
                                    context, question, fb
                                )
                                if not (isinstance(temp.get('answer'), str) and temp['answer'].startswith("Error:")):
                                    result = temp
                                    used_display_name = f"{model_display_name} (fallback)"
                                    break
                        inference_time = time.time() - start_time
                        
                        comparison_results.append({
                            'Model': used_display_name,
                            'Answer': result['answer'],
                            'Confidence': result['score'],
                            'Inference Time (s)': round(inference_time, 3)
                        })
                    except Exception as e:
                        comparison_results.append({
                            'Model': model_display_name,
                            'Answer': f"Error: {str(e)}",
                            'Confidence': 0.0,
                            'Inference Time (s)': 0.0
                        })
                
                # Complete progress bar
                progress_bar.progress(1.0, text="Comparison complete!")
                
                # Display results table
                st.subheader("Comparison Results")
                for result in comparison_results:
                    with st.expander(f"Model: {result['Model']}"):
                        st.write(f"**Answer:** {result['Answer']}")
                        st.write(f"**Confidence:** {result['Confidence']:.3f}")
                        st.write(f"**Inference Time:** {result['Inference Time (s)']:.3f}s")
                        
                        # Simple confidence bar
                        st.progress(result['Confidence'])
                
                # Simple summary
                st.subheader("Summary")
                best_confidence = max(comparison_results, key=lambda x: x['Confidence'])
                fastest_model = min(comparison_results, key=lambda x: x['Inference Time (s)'])
                
                st.write(f"**Highest Confidence:** {best_confidence['Model']} ({best_confidence['Confidence']:.3f})")
                st.write(f"**Fastest Model:** {fastest_model['Model']} ({fastest_model['Inference Time (s)']:.3f}s)")
        else:
            st.warning("Please enter a context and question first in the Question Answering tab.")

with tab3:
    st.header("Model Evaluation Metrics")
    
    # Upload evaluation dataset
    uploaded_file = st.file_uploader(
        "Upload evaluation dataset (JSON format)",
        type=['json'],
        help="Upload a dataset in SQuAD format for evaluation"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_upload_{int(time.time())}.json"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run evaluation
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating model..."):
                try:
                    # Get evaluation results
                    eval_results = st.session_state.evaluator.evaluate_model_on_dataset(
                        temp_path, model_name
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Exact Match", f"{eval_results['exact_match']:.3f}")
                    with col2:
                        st.metric("F1 Score", f"{eval_results['f1_score']:.3f}")
                    with col3:
                        st.metric("Total Questions", eval_results['total_questions'])
                    
                    # Show sample predictions if available
                    if 'sample_predictions' in eval_results and eval_results['sample_predictions']:
                        st.subheader("Sample Predictions")
                        for i, pred in enumerate(eval_results['sample_predictions'][:5]):
                            with st.expander(f"Prediction {i+1}"):
                                st.write(f"**Predicted:** {pred.get('prediction', 'N/A')}")
                                st.write(f"**Expected:** {pred.get('ground_truth', 'N/A')}")
                                st.write(f"**Match:** {'✓' if pred.get('exact_match', 0) == 1 else '✗'}")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    else:
        st.info("Upload an evaluation dataset to see detailed metrics.")
        
        # Show real-time evaluation option
        st.subheader("Quick Evaluation")
        if st.button("Evaluate on Sample SQuAD Data"):
            with st.spinner("Loading SQuAD dataset and evaluating..."):
                try:
                    # Use the data loader to get SQuAD data
                    from utils.data_loader import DataLoader
                    data_loader = DataLoader()
                    
                    # Load a small sample of SQuAD validation data
                    squad_data = data_loader.load_squad_v1("validation", max_samples=50)
                    
                    if squad_data:
                        # Create temporary dataset file
                        temp_squad_path = "temp_squad_eval.json"
                        
                        # Group by context to create proper SQuAD format
                        context_groups = {}
                        for item in squad_data:
                            context = item['context']
                            if context not in context_groups:
                                context_groups[context] = []
                            
                            context_groups[context].append({
                                'id': item['id'],
                                'question': item['question'],
                                'answers': [{'text': ans, 'answer_start': 0} for ans in item['answers']]
                            })
                        
                        # Create SQuAD format
                        squad_format = {
                            'version': '1.1',
                            'data': [{
                                'title': 'SQuAD Validation Sample',
                                'paragraphs': [
                                    {'context': context, 'qas': qas}
                                    for context, qas in context_groups.items()
                                ]
                            }]
                        }
                        
                        with open(temp_squad_path, 'w', encoding='utf-8') as f:
                            json.dump(squad_format, f, ensure_ascii=False, indent=2)
                        
                        # Run evaluation
                        eval_results = st.session_state.evaluator.evaluate_model_on_dataset(
                            temp_squad_path, model_name
                        )
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Exact Match", f"{eval_results['exact_match']:.3f}")
                        with col2:
                            st.metric("F1 Score", f"{eval_results['f1_score']:.3f}")
                        with col3:
                            st.metric("Total Questions", eval_results['total_questions'])
                        
                        # Clean up
                        os.remove(temp_squad_path)
                        
                    else:
                        st.error("Could not load SQuAD data")
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

with tab4:
    st.header("Sample Questions")
    st.info("These are pre-loaded example questions to help you test the system. You can use them as templates or modify them for your own questions.")
    
    # English samples
    st.subheader("🇺🇸 English Samples")
    
    english_samples = [
        {
            "context": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest.",
            "question": "How much area does the Amazon basin cover?"
        },
        {
            "context": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "question": "What is artificial intelligence?"
        }
    ]
    
    for i, sample in enumerate(english_samples):
        with st.expander(f"English Sample {i+1}"):
            if st.button(f"Load English Sample {i+1}", key=f"en_{i}"):
                st.session_state.sample_context = sample["context"]
                st.session_state.sample_question = sample["question"]
                st.rerun()
            
            st.write("**Context:**", sample["context"])
            st.write("**Question:**", sample["question"])
    
    # Arabic samples
    st.subheader("🇸🇦 Arabic Samples")
    
    arabic_samples = [
        {
            "context": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. تقع على ضفاف نهر النيل في شمال مصر، وهي واحدة من أكبر المدن في أفريقيا والشرق الأوسط. يبلغ عدد سكان القاهرة الكبرى حوالي 20 مليون نسمة.",
            "question": "ما هي عاصمة مصر؟"
        },
        {
            "context": "الذكاء الاصطناعي هو محاكاة عمليات الذكاء البشري بواسطة الآلات، وخاصة أنظمة الحاسوب. تشمل هذه العمليات التعلم واكتساب المعلومات وقواعد استخدام المعلومات والتفكير المنطقي والتصحيح الذاتي.",
            "question": "ما هو الذكاء الاصطناعي؟"
        }
    ]
    
    for i, sample in enumerate(arabic_samples):
        with st.expander(f"Arabic Sample {i+1}"):
            if st.button(f"Load Arabic Sample {i+1}", key=f"ar_{i}"):
                st.session_state.sample_context = sample["context"]
                st.session_state.sample_question = sample["question"]
                st.rerun()
            
            st.write("**Context:**", sample["context"])
            st.write("**Question:**", sample["question"])
    
    # Load sample button effect
    if hasattr(st.session_state, 'sample_context') and hasattr(st.session_state, 'sample_question'):
        st.success("Sample loaded! Go to the Question Answering tab to see the results.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit and Hugging Face Transformers</p>
        <p>Supports: BERT, DistilBERT, Multilingual BERT</p>
    </div>
    """, 
    unsafe_allow_html=True
)
