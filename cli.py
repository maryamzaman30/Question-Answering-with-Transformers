#!/usr/bin/env python3
"""
Command Line Interface for Multilingual Question Answering System

This CLI provides an interactive command-line interface for the multilingual QA system,
supporting both English and Arabic question answering using transformer models.
"""

import argparse
import sys
import os
import json
import time
from typing import Dict, Any, List

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Silence TensorFlow oneDNN logs and disable TF usage by Transformers in this CLI context
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Instruct Transformers to skip TensorFlow entirely in CLI runs
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from models.model_handler import ModelHandler
from utils.language_detector import LanguageDetector
from utils.evaluation import QAEvaluator
from utils.data_loader import DataLoader


class QACLIInterface:
    """Command Line Interface for Question Answering System"""
    
    def __init__(self):
        self.model_handler = ModelHandler()
        self.language_detector = LanguageDetector()
        self.evaluator = QAEvaluator()
        self.data_loader = DataLoader()
        
        # Available models (aligned with app.py defaults)
        self.available_models = {
            "bert-en": "bert-large-uncased-whole-word-masking-finetuned-squad",
            "distilbert-en": "distilbert-base-uncased-distilled-squad",
            "mbert": "bert-base-multilingual-cased"
        }
        
        print("🤖 Multilingual Question Answering CLI Interface")
        print("=" * 60)
    
    def display_available_models(self):
        """Display available models"""
        print("\nAvailable Models:")
        print("-" * 40)
        for short_name, full_name in self.available_models.items():
            print(f"  {short_name:15} : {full_name}")
    
    def get_recommended_model(self, text: str) -> str:
        """Get recommended model based on detected language"""
        detected_lang = self.language_detector.detect_language(text)
        
        if detected_lang == "Arabic":
            # Prefer a multilingual checkpoint fine-tuned on SQuAD for better Arabic performance
            return "salti/bert-base-multilingual-cased-finetuned-squad"
        elif detected_lang == "English":
            return self.available_models["bert-en"]
        else:
            return self.available_models["mbert"]  # Multilingual fallback
    
    def ask_question(self, context: str, question: str, model_name: str = None, 
                    auto_detect: bool = True, max_answer_length: int = 30) -> Dict[str, Any]:
        """Process a single question"""
        
        # Auto-detect language and recommend model if not specified
        text_for_detection = f"{question} {context}"
        detected_lang = self.language_detector.detect_language(text_for_detection)
        if auto_detect and model_name is None:
            model_name = self.get_recommended_model(text_for_detection)
            print(f"🔍 Detected language: {detected_lang}")
            print(f"📝 Using model: {model_name.split('/')[-1]}")
        elif model_name is None:
            model_name = self.available_models["mbert"]  # Default fallback
        
        # Get answer
        start_time = time.time()
        try:
            result = self.model_handler.get_answer(context, question, model_name, max_answer_length)
            inference_time = time.time() - start_time
            
            # Simple fallback for Arabic if confidence is extremely low
            if auto_detect and detected_lang == "Arabic" and result.get("score", 0.0) < 0.05:
                fallback_model = "salti/bert-base-multilingual-cased-finetuned-squad"
                if model_name != fallback_model:
                    print("⚠️ Low confidence detected. Trying Arabic fallback model (mBERT fine-tuned)...")
                    start_fb = time.time()
                    fb_res = self.model_handler.get_answer(context, question, fallback_model, max_answer_length)
                    # Use fallback if it yields higher confidence
                    if fb_res.get("score", 0.0) >= result.get("score", 0.0):
                        result = fb_res
                        inference_time = time.time() - start_fb
                        model_name = fallback_model
            
            return {
                "success": True,
                "answer": result["answer"],
                "confidence": result["score"],
                "start_pos": result["start"],
                "end_pos": result["end"],
                "inference_time": inference_time,
                "model_used": model_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "inference_time": time.time() - start_time,
                "model_used": model_name
            }
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n🎯 Interactive Question Answering Mode")
        print("Type 'quit' to exit, 'help' for commands, 'models' to see available models")
        print("=" * 60)
        
        current_context = ""
        current_model = None
        auto_detect = True
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif command.lower() == 'help':
                    self.display_help()
                
                elif command.lower() == 'models':
                    self.display_available_models()
                
                elif command.lower().startswith('set_model '):
                    model_key = command.split(' ', 1)[1]
                    if model_key in self.available_models:
                        current_model = self.available_models[model_key]
                        auto_detect = False
                        print(f"✅ Model set to: {current_model.split('/')[-1]}")
                    else:
                        print(f"❌ Unknown model: {model_key}")
                        self.display_available_models()
                
                elif command.lower() == 'auto_detect':
                    auto_detect = True
                    current_model = None
                    print("✅ Auto-detection enabled")
                
                elif command.lower().startswith('context '):
                    current_context = command[8:].strip()
                    print(f"✅ Context set ({len(current_context)} characters)")
                    if self.language_detector.detect_language(current_context) == "Arabic":
                        print("🔤 Arabic text detected")
                    else:
                        print("🔤 English text detected")
                
                elif command.lower() == 'show_context':
                    if current_context:
                        print(f"📄 Current context:\n{current_context}")
                    else:
                        print("❌ No context set. Use 'context <your_context>' to set one.")
                
                elif current_context and command:
                    # Treat as question
                    question = command
                    print(f"\n❓ Question: {question}")
                    print("🤔 Processing...")
                    
                    result = self.ask_question(
                        current_context, 
                        question, 
                        current_model, 
                        auto_detect
                    )
                    
                    if result["success"]:
                        print(f"\n✅ Answer: {result['answer']}")
                        print(f"🎯 Confidence: {result['confidence']:.3f}")
                        print(f"⚡ Inference time: {result['inference_time']:.3f}s")
                        print(f"🤖 Model: {result['model_used'].split('/')[-1]}")
                        
                        # Show confidence indicator
                        if result['confidence'] > 0.7:
                            print("🟢 High confidence")
                        elif result['confidence'] > 0.3:
                            print("🟡 Medium confidence")
                        else:
                            print("🔴 Low confidence")
                    else:
                        print(f"❌ Error: {result['error']}")
                
                else:
                    print("❌ Please set a context first using 'context <your_context>'")
                    print("💡 Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    def display_help(self):
        """Display help information"""
        help_text = """
🆘 Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context Management:
  context <text>          Set the context/passage for Q&A
  show_context            Display current context

Model Management:
  models                  Show available models
  set_model <model_key>   Set specific model (e.g., bert-en, distilbert-en, mbert)
  auto_detect             Enable automatic language detection

Question Answering:
  <your_question>         Ask a question about the current context

General:
  help                    Show this help message
  quit/exit/q            Exit the program

Examples:
  context The Amazon rainforest covers most of the Amazon basin...
  What is the Amazon rainforest?
  
  context القاهرة هي عاصمة مصر...
  ما هي عاصمة مصر؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        print(help_text)
    
    def batch_mode(self, input_file: str, output_file: str = None, model_name: str = None):
        """Process questions from a file"""
        print(f"\n📁 Batch Processing Mode: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            
            for i, item in enumerate(data):
                context = item.get('context', '')
                question = item.get('question', '')
                expected_answer = item.get('answer', None)
                
                print(f"Processing question {i+1}/{len(data)}...")
                
                result = self.ask_question(context, question, model_name)
                
                # Add evaluation metrics if expected answer is provided
                if expected_answer and result["success"]:
                    exact_match = self.evaluator.exact_match_score(result["answer"], expected_answer)
                    f1_score = self.evaluator.f1_score(result["answer"], expected_answer)
                    result["exact_match"] = exact_match
                    result["f1_score"] = f1_score
                    result["expected_answer"] = expected_answer
                
                result["question"] = question
                result["context"] = context
                results.append(result)
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to: {output_file}")
            
            # Display summary
            successful = sum(1 for r in results if r["success"])
            print(f"\n📊 Batch Processing Summary:")
            print(f"  Total questions: {len(results)}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {len(results) - successful}")
            
            if any("exact_match" in r for r in results):
                avg_em = sum(r.get("exact_match", 0) for r in results) / len(results)
                avg_f1 = sum(r.get("f1_score", 0) for r in results) / len(results)
                print(f"  Average Exact Match: {avg_em:.3f}")
                print(f"  Average F1 Score: {avg_f1:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing batch file: {e}")
    
    def compare_models(self, context: str, question: str, models: List[str] = None):
        """Compare multiple models on the same question"""
        if models is None:
            models = ["bert-en", "distilbert-en", "mbert"]
        
        print(f"\n🔍 Comparing Models")
        print("=" * 50)
        print(f"Question: {question}")
        print(f"Context: {context[:100]}{'...' if len(context) > 100 else ''}")
        print()
        
        results = []
        
        for model_key in models:
            if model_key in self.available_models:
                model_name = self.available_models[model_key]
                print(f"Testing {model_key}...")
                
                result = self.ask_question(context, question, model_name, auto_detect=False)
                result["model_key"] = model_key
                results.append(result)
        
        # Display comparison
        print("\n📊 Comparison Results:")
        print("-" * 70)
        print(f"{'Model':<15} {'Answer':<30} {'Confidence':<12} {'Time':<8}")
        print("-" * 70)
        
        for result in results:
            if result["success"]:
                answer = result["answer"][:28] + "..." if len(result["answer"]) > 30 else result["answer"]
                print(f"{result['model_key']:<15} {answer:<30} {result['confidence']:<12.3f} {result['inference_time']:<8.3f}")
            else:
                print(f"{result['model_key']:<15} {'ERROR':<30} {'N/A':<12} {'N/A':<8}")
        
        return results


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Multilingual Question Answering CLI")
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    parser.add_argument('--context', '-c', type=str,
                       help='Context/passage for question answering')
    
    parser.add_argument('--question', '-q', type=str,
                       help='Question to ask')
    
    parser.add_argument('--model', '-m', type=str,
                       help='Model to use (e.g., bert-en, distilbert-en, mbert)')
    
    parser.add_argument('--batch', '-b', type=str,
                       help='JSON file with questions for batch processing')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for batch results')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models on the same question')
    
    parser.add_argument('--models', action='store_true',
                       help='List available models')
    
    parser.add_argument('--auto-detect', action='store_true', default=True,
                       help='Enable automatic language detection')
    
    args = parser.parse_args()
    
    # Initialize CLI interface
    cli = QACLIInterface()
    
    try:
        if args.models:
            cli.display_available_models()
        
        elif args.interactive:
            cli.interactive_mode()
        
        elif args.batch:
            model_name = cli.available_models.get(args.model) if args.model else None
            cli.batch_mode(args.batch, args.output, model_name)
        
        elif args.context and args.question:
            if args.compare:
                cli.compare_models(args.context, args.question)
            else:
                model_name = cli.available_models.get(args.model) if args.model else None
                
                print("🤔 Processing question...")
                result = cli.ask_question(args.context, args.question, model_name, args.auto_detect)
                
                if result["success"]:
                    print(f"\n✅ Answer: {result['answer']}")
                    print(f"🎯 Confidence: {result['confidence']:.3f}")
                    print(f"⚡ Inference time: {result['inference_time']:.3f}s")
                    print(f"🤖 Model: {result['model_used'].split('/')[-1]}")
                else:
                    print(f"❌ Error: {result['error']}")
        
        else:
            print("🚀 Welcome to the Multilingual QA CLI!")
            print("\nQuick start options:")
            print("  --interactive      : Start interactive mode")
            print("  --models          : List available models")
            print("  --help            : Show all options")
            print("\nExample:")
            print('  python cli.py --context "The Amazon rainforest..." --question "What is the Amazon?"')
            print('  python cli.py --interactive')
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
