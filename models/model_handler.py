import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import logging
from typing import Dict, Any
import os

class ModelHandler:
    """Handles loading and inference for various QA models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_name: str) -> None:
        """Load a model and tokenizer if not already loaded"""
        if model_name not in self.models:
            try:
                self.logger.info(f"Loading model: {model_name}")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                
                # Move model to device
                model.to(self.device)
                
                # Create pipeline
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Store components
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                self.pipelines[model_name] = qa_pipeline
                
                self.logger.info(f"Successfully loaded model: {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {str(e)}")
                raise e
    
    def get_answer(self, context: str, question: str, model_name: str, max_answer_length: int = 30) -> Dict[str, Any]:
        """Get answer for a question given context using specified model"""
        try:
            # Load model if not already loaded
            if model_name not in self.pipelines:
                self.load_model(model_name)
            
            # Get answer using pipeline
            result = self.pipelines[model_name](
                question=question,
                context=context,
                max_answer_len=max_answer_length
            )
            
            return {
                'answer': result['answer'],
                'score': result['score'],
                'start': result['start'],
                'end': result['end']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting answer: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'score': 0.0,
                'start': 0,
                'end': 0
            }
    
    def get_detailed_prediction(self, context: str, question: str, model_name: str) -> Dict[str, Any]:
        """Get detailed prediction with token-level information"""
        try:
            # Load model if not already loaded
            if model_name not in self.models:
                self.load_model(model_name)
            
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Tokenize inputs
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get start and end logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Get the most likely start and end positions
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            
            # Calculate confidence scores
            start_score = torch.softmax(start_logits, dim=1)[0][start_idx].item()
            end_score = torch.softmax(end_logits, dim=1)[0][end_idx].item()
            confidence = (start_score + end_score) / 2
            
            # Extract answer tokens
            input_ids = inputs['input_ids'][0]
            answer_tokens = input_ids[start_idx:end_idx + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_score': start_score,
                'end_score': end_score,
                'input_tokens': tokenizer.convert_ids_to_tokens(input_ids),
                'answer_tokens': tokenizer.convert_ids_to_tokens(answer_tokens)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting detailed prediction: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'start_idx': 0,
                'end_idx': 0,
                'start_score': 0.0,
                'end_score': 0.0,
                'input_tokens': [],
                'answer_tokens': []
            }
    
    def compare_models(self, context: str, question: str, model_names: list, max_answer_length: int = 30) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models on the same question-context pair"""
        results = {}
        
        for model_name in model_names:
            try:
                result = self.get_answer(context, question, model_name, max_answer_length)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {
                    'answer': f"Error: {str(e)}",
                    'score': 0.0,
                    'start': 0,
                    'end': 0
                }
        
        return results
    
    def clear_cache(self):
        """Clear all loaded models to free memory"""
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Model cache cleared")

    def _prepare_train_features(self, examples, tokenizer: AutoTokenizer, max_length: int, doc_stride: int):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            context_index = 1
            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != context_index:
                token_start_index += 1
            token_end_index = len(sequence_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != context_index:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    def train_on_squad(self,
                        model_name_or_path: str,
                        output_dir: str,
                        num_train_epochs: float = 1.0,
                        per_device_train_batch_size: int = 8,
                        learning_rate: float = 3e-5,
                        weight_decay: float = 0.01,
                        max_length: int = 384,
                        doc_stride: int = 128,
                        eval_samples: int = 200) -> Dict[str, Any]:
        """Fine-tune a QA model on SQuAD v1.1 and return EM/F1 on a validation subset."""
        # Lazy import training utilities to avoid pulling TF/Keras at module import time
        from transformers import Trainer, TrainingArguments, default_data_collator
        self.logger.info("Loading SQuAD dataset...")
        raw_datasets = load_dataset("squad")

        self.logger.info(f"Loading tokenizer and model: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

        def _prep(examples):
            return self._prepare_train_features(examples, tokenizer, max_length, doc_stride)

        self.logger.info("Tokenizing training data...")
        train_dataset = raw_datasets["train"].map(
            _prep,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            save_total_limit=2,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        self.logger.info("Starting fine-tuning on SQuAD v1.1...")
        trainer.train()

        self.logger.info(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Quick evaluation on a subset of validation
        from utils.evaluation import QAEvaluator
        eval_pipe = pipeline(
            "question-answering",
            model=output_dir,
            tokenizer=output_dir,
            device=0 if torch.cuda.is_available() else -1,
        )

        val_ds = raw_datasets["validation"]
        n = min(eval_samples, len(val_ds))
        evaluator = QAEvaluator()
        predictions = []
        ground_truths = []
        for i in range(n):
            item = val_ds[i]
            question = item["question"]
            context = item["context"]
            gt = item["answers"]["text"][0] if len(item["answers"]["text"]) else ""
            try:
                pred = eval_pipe(question=question, context=context)["answer"]
            except Exception:
                pred = ""
            predictions.append(pred)
            ground_truths.append(gt)

        metrics = evaluator.evaluate_predictions(predictions, ground_truths)
        metrics["output_dir"] = output_dir
        self.logger.info(f"SQuAD eval subset — EM: {metrics['exact_match']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
