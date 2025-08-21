import json
import pandas as pd
from typing import Dict, List, Any, Union
import re
import string
from collections import Counter
import numpy as np
from models.model_handler import ModelHandler
import logging

class QAEvaluator:
    """Evaluation utilities for question answering models"""
    
    def __init__(self):
        self.model_handler = ModelHandler()
        self.logger = logging.getLogger(__name__)
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate exact match score"""
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score based on token overlap"""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return float(prediction_tokens == ground_truth_tokens)
        
        common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_predictions(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a list of predictions against ground truths"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        exact_matches = []
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            exact_matches.append(self.exact_match_score(pred, gt))
            f1_scores.append(self.f1_score(pred, gt))
        
        return {
            'exact_match': np.mean(exact_matches),
            'f1_score': np.mean(f1_scores),
            'total_questions': len(predictions)
        }
    
    def evaluate_model_on_dataset(self, dataset_file, model_name: str, max_samples: int = 100) -> Dict[str, Any]:
        """Evaluate a model on a dataset"""
        try:
            # Load dataset
            if hasattr(dataset_file, 'read'):
                dataset = json.load(dataset_file)
            else:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            
            # Extract questions and contexts
            questions = []
            contexts = []
            ground_truths = []
            
            # Handle SQuAD format
            if 'data' in dataset:
                for article in dataset['data'][:max_samples]:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            questions.append(qa['question'])
                            contexts.append(context)
                            # Get first answer as ground truth
                            if qa['answers']:
                                ground_truths.append(qa['answers'][0]['text'])
                            else:
                                ground_truths.append("")
            else:
                # Handle simple format
                for item in dataset[:max_samples]:
                    questions.append(item['question'])
                    contexts.append(item['context'])
                    ground_truths.append(item['answer'])
            
            # Limit to max_samples
            questions = questions[:max_samples]
            contexts = contexts[:max_samples]
            ground_truths = ground_truths[:max_samples]
            
            # Get predictions
            predictions = []
            prediction_details = []
            
            for i, (question, context) in enumerate(zip(questions, contexts)):
                try:
                    result = self.model_handler.get_answer(context, question, model_name)
                    predictions.append(result['answer'])
                    
                    prediction_details.append({
                        'question': question,
                        'context': context[:200] + "..." if len(context) > 200 else context,
                        'predicted_answer': result['answer'],
                        'ground_truth': ground_truths[i],
                        'confidence': result['score'],
                        'exact_match': self.exact_match_score(result['answer'], ground_truths[i]),
                        'f1_score': self.f1_score(result['answer'], ground_truths[i])
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing question {i}: {str(e)}")
                    predictions.append("")
                    prediction_details.append({
                        'question': question,
                        'context': context[:200] + "..." if len(context) > 200 else context,
                        'predicted_answer': "",
                        'ground_truth': ground_truths[i],
                        'confidence': 0.0,
                        'exact_match': 0.0,
                        'f1_score': 0.0
                    })
            
            # Calculate overall metrics
            evaluation_results = self.evaluate_predictions(predictions, ground_truths)
            evaluation_results['predictions'] = prediction_details
            evaluation_results['model_name'] = model_name
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'exact_match': 0.0,
                'f1_score': 0.0,
                'total_questions': 0,
                'error': str(e)
            }
    
    def compare_models_on_dataset(self, dataset_file, model_names: List[str], max_samples: int = 50) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models on the same dataset"""
        results = {}
        
        for model_name in model_names:
            self.logger.info(f"Evaluating model: {model_name}")
            results[model_name] = self.evaluate_model_on_dataset(dataset_file, model_name, max_samples)
        
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a formatted evaluation report"""
        report = f"""
# Evaluation Report

## Model: {evaluation_results.get('model_name', 'Unknown')}

### Overall Metrics
- **Exact Match**: {evaluation_results['exact_match']:.3f}
- **F1 Score**: {evaluation_results['f1_score']:.3f}
- **Total Questions**: {evaluation_results['total_questions']}

### Performance Analysis
- **High Confidence Predictions (>0.7)**: {sum(1 for p in evaluation_results.get('predictions', []) if p['confidence'] > 0.7)} / {evaluation_results['total_questions']}
- **Perfect Matches**: {sum(1 for p in evaluation_results.get('predictions', []) if p['exact_match'] == 1.0)} / {evaluation_results['total_questions']}
- **Good F1 Scores (>0.8)**: {sum(1 for p in evaluation_results.get('predictions', []) if p['f1_score'] > 0.8)} / {evaluation_results['total_questions']}

### Sample Predictions
"""
        
        # Add sample predictions
        if 'predictions' in evaluation_results:
            for i, pred in enumerate(evaluation_results['predictions'][:5]):
                report += f"""
#### Question {i+1}
- **Question**: {pred['question']}
- **Ground Truth**: {pred['ground_truth']}
- **Prediction**: {pred['predicted_answer']}
- **Exact Match**: {pred['exact_match']}
- **F1 Score**: {pred['f1_score']:.3f}
- **Confidence**: {pred['confidence']:.3f}
"""
        
        return report
