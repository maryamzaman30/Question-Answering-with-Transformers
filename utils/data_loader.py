import json
import pandas as pd
from typing import Dict, List, Any, Optional
import requests
import os
from datasets import load_dataset
import logging

class DataLoader:
    """Handles loading and preprocessing of QA datasets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_squad_v1(self, split: str = "train", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load SQuAD v1.1 dataset"""
        try:
            self.logger.info(f"Loading SQuAD v1.1 {split} split...")
            dataset = load_dataset("squad", split=split)
            
            qa_pairs = []
            for i, example in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                qa_pairs.append({
                    'id': example['id'],
                    'question': example['question'],
                    'context': example['context'],
                    'answers': example['answers']['text'],
                    'answer_starts': example['answers']['answer_start']
                })
            
            self.logger.info(f"Loaded {len(qa_pairs)} examples from SQuAD v1.1")
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error loading SQuAD v1.1: {str(e)}")
            return []
    
    def load_arabic_squad(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load Arabic SQuAD dataset (ARCD)"""
        try:
            self.logger.info("Loading Arabic SQuAD (ARCD)...")
            dataset = load_dataset("arcd", split="train")
            
            qa_pairs = []
            for i, example in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                
                qa_pairs.append({
                    'id': example.get('id', f'arcd_{i}'),
                    'question': example['question'],
                    'context': example['context'],
                    'answers': example['answers']['text'],
                    'answer_starts': example['answers']['answer_start']
                })
            
            self.logger.info(f"Loaded {len(qa_pairs)} examples from Arabic SQuAD")
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error loading Arabic SQuAD: {str(e)}")
            return []
    
    def load_custom_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load custom dataset from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_pairs = []
            
            # Handle SQuAD format
            if 'data' in data:
                for article in data['data']:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            qa_pairs.append({
                                'id': qa['id'],
                                'question': qa['question'],
                                'context': context,
                                'answers': [answer['text'] for answer in qa['answers']],
                                'answer_starts': [answer['answer_start'] for answer in qa['answers']]
                            })
            
            # Handle simple format
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    qa_pairs.append({
                        'id': item.get('id', f'custom_{i}'),
                        'question': item['question'],
                        'context': item['context'],
                        'answers': [item['answer']] if 'answer' in item else item.get('answers', []),
                        'answer_starts': item.get('answer_starts', [0])
                    })
            
            self.logger.info(f"Loaded {len(qa_pairs)} examples from custom dataset")
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error loading custom dataset: {str(e)}")
            return []
    
    def create_sample_english_dataset(self) -> List[Dict[str, Any]]:
        """Create sample English dataset for testing"""
        return [
            {
                'id': 'sample_en_1',
                'question': 'What is the capital of France?',
                'context': 'France is a country in Western Europe. Its capital and largest city is Paris, which is located in the north-central part of the country.',
                'answers': ['Paris'],
                'answer_starts': [72]
            },
            {
                'id': 'sample_en_2',
                'question': 'How many legs does a spider have?',
                'context': 'Spiders are arachnids, not insects. They have eight legs, unlike insects which have six legs. Spiders also have two body segments.',
                'answers': ['eight'],
                'answer_starts': [51]
            },
            {
                'id': 'sample_en_3',
                'question': 'When was the Declaration of Independence signed?',
                'context': 'The Declaration of Independence was signed on July 4, 1776, in Philadelphia. This document declared the American colonies independent from British rule.',
                'answers': ['July 4, 1776'],
                'answer_starts': [49]
            }
        ]
    
    def create_sample_arabic_dataset(self) -> List[Dict[str, Any]]:
        """Create sample Arabic dataset for testing"""
        return [
            {
                'id': 'sample_ar_1',
                'question': 'ما هي عاصمة مصر؟',
                'context': 'مصر دولة عربية تقع في شمال أفريقيا. عاصمتها القاهرة وهي أكبر مدنها. تقع القاهرة على ضفاف نهر النيل.',
                'answers': ['القاهرة'],
                'answer_starts': [41]
            },
            {
                'id': 'sample_ar_2',
                'question': 'كم عدد أيام السنة؟',
                'context': 'السنة الميلادية تحتوي على ثلاثمائة وخمسة وستين يوماً في السنة العادية، وثلاثمائة وستة وستين يوماً في السنة الكبيسة.',
                'answers': ['ثلاثمائة وخمسة وستين'],
                'answer_starts': [29]
            }
        ]
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> str:
        """Save dataset to JSON file"""
        file_path = os.path.join(self.cache_dir, filename)
        
        # Convert to SQuAD format
        squad_format = {
            'version': '1.1',
            'data': [{
                'title': 'Generated Dataset',
                'paragraphs': []
            }]
        }
        
        # Group by context
        context_groups = {}
        for item in dataset:
            context = item['context']
            if context not in context_groups:
                context_groups[context] = []
            
            context_groups[context].append({
                'id': item['id'],
                'question': item['question'],
                'answers': [
                    {
                        'text': answer,
                        'answer_start': start
                    }
                    for answer, start in zip(item['answers'], item['answer_starts'])
                ]
            })
        
        # Add to SQuAD format
        for context, qas in context_groups.items():
            squad_format['data'][0]['paragraphs'].append({
                'context': context,
                'qas': qas
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(squad_format, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Dataset saved to {file_path}")
        return file_path
    
    def get_dataset_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        if not dataset:
            return {}
        
        questions = [item['question'] for item in dataset]
        contexts = [item['context'] for item in dataset]
        answers = [answer for item in dataset for answer in item['answers']]
        
        stats = {
            'total_examples': len(dataset),
            'avg_question_length': sum(len(q.split()) for q in questions) / len(questions),
            'avg_context_length': sum(len(c.split()) for c in contexts) / len(contexts),
            'avg_answer_length': sum(len(a.split()) for a in answers) / len(answers) if answers else 0,
            'unique_contexts': len(set(contexts)),
            'total_answers': len(answers)
        }
        
        return stats
