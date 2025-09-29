"""
Speech-to-text processing and word embedding for Jerome Powell speech analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModel
import re
from collections import Counter
import pickle
from pathlib import Path


class SpeechToTextProcessor:
    """
    Convert Jerome Powell speech audio to text and extract beginning words/phrases
    """
    
    def __init__(self, language: str = "en-US"):
        self.recognizer = sr.Recognizer()
        self.language = language
        
        # Key phrases that often start Powell's responses
        self.key_beginning_phrases = [
            "well", "so", "let me", "i think", "we believe", "the committee",
            "as i", "our view", "we see", "looking ahead", "going forward",
            "at this point", "right now", "currently", "we expect", "we anticipate"
        ]
    
    def audio_to_text(self, audio_path: str) -> str:
        """
        Convert audio file to text using speech recognition
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = self.recognizer.record(source)
            
            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data, language=self.language)
            return text.lower()
            
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return ""
    
    def extract_beginning_words(self, text: str, num_words: int = 10) -> List[str]:
        """
        Extract beginning words from speech text
        
        Args:
            text: Transcribed speech text
            num_words: Number of beginning words to extract
            
        Returns:
            List of beginning words
        """
        # Clean and tokenize text
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()
        
        # Extract first num_words
        beginning_words = words[:num_words] if len(words) >= num_words else words
        
        # Pad with empty strings if needed
        while len(beginning_words) < num_words:
            beginning_words.append("")
        
        return beginning_words
    
    def detect_key_phrases(self, text: str) -> Dict[str, bool]:
        """
        Detect presence of key beginning phrases that indicate tone
        
        Args:
            text: Transcribed speech text
            
        Returns:
            Dictionary mapping phrases to presence indicators
        """
        text_lower = text.lower()
        phrase_indicators = {}
        
        for phrase in self.key_beginning_phrases:
            phrase_indicators[phrase] = phrase in text_lower
        
        return phrase_indicators
    
    def extract_sentence_starters(self, text: str, max_sentences: int = 5) -> List[str]:
        """
        Extract the beginning of first few sentences
        
        Args:
            text: Transcribed speech text
            max_sentences: Maximum number of sentences to process
            
        Returns:
            List of sentence beginnings
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_starters = []
        
        for i, sentence in enumerate(sentences[:max_sentences]):
            if sentence.strip():
                # Get first 5 words of each sentence
                words = sentence.strip().split()[:5]
                starter = " ".join(words)
                sentence_starters.append(starter.lower())
        
        return sentence_starters


class WordEmbeddingProcessor:
    """
    Process words and phrases into embeddings for the HRM model
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.model.eval()
        
        # Vocabulary for Powell-specific terms
        self.fed_vocabulary = {
            # Policy terms
            "hawkish": ["aggressive", "tighten", "raise", "hike", "restrictive"],
            "dovish": ["accommodative", "lower", "cut", "reduce", "supportive"],
            "neutral": ["maintain", "hold", "steady", "appropriate", "gradual"],
            
            # Economic indicators
            "inflation": ["prices", "pce", "cpi", "core", "headline"],
            "employment": ["jobs", "unemployment", "labor", "payrolls", "wages"],
            "growth": ["gdp", "economy", "expansion", "recovery", "output"]
        }
    
    def embed_words(self, words: List[str]) -> torch.Tensor:
        """
        Convert list of words to embeddings
        
        Args:
            words: List of words to embed
            
        Returns:
            Embedding tensor [num_words, embedding_dim]
        """
        if not words or all(word == "" for word in words):
            # Return zero embeddings for empty input
            return torch.zeros(len(words) if words else 1, self.model.config.hidden_size)
        
        # Join words into text
        text = " ".join(word for word in words if word)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
        
        return embeddings.squeeze(0)  # [hidden_size]
    
    def embed_phrase_sequence(self, phrases: List[str]) -> torch.Tensor:
        """
        Embed a sequence of phrases (e.g., sentence starters)
        
        Args:
            phrases: List of phrases to embed
            
        Returns:
            Embedding tensor [num_phrases, embedding_dim]
        """
        embeddings = []
        
        for phrase in phrases:
            if phrase.strip():
                inputs = self.tokenizer(
                    phrase,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    phrase_embedding = outputs.last_hidden_state[:, 0, :]
                    embeddings.append(phrase_embedding.squeeze(0))
            else:
                # Zero embedding for empty phrases
                embeddings.append(torch.zeros(self.model.config.hidden_size))
        
        return torch.stack(embeddings) if embeddings else torch.zeros(1, self.model.config.hidden_size)
    
    def extract_fed_sentiment_features(self, text: str) -> torch.Tensor:
        """
        Extract Fed-specific sentiment features from text
        
        Args:
            text: Speech text
            
        Returns:
            Feature vector with Fed sentiment indicators
        """
        text_lower = text.lower()
        features = []
        
        # Count occurrences of different sentiment categories
        for category, terms in self.fed_vocabulary.items():
            category_count = sum(1 for term in terms if term in text_lower)
            features.append(category_count)
        
        # Add ratios and normalized counts
        total_words = len(text_lower.split())
        if total_words > 0:
            features.extend([count / total_words for count in features])
        else:
            features.extend([0.0] * len(self.fed_vocabulary))
        
        return torch.tensor(features, dtype=torch.float32)


class BeginningWordsProcessor:
    """
    Main processor for extracting and embedding beginning words from Powell speeches
    """
    
    def __init__(self, embedding_model: str = "bert-base-uncased"):
        self.speech_processor = SpeechToTextProcessor()
        self.embedding_processor = WordEmbeddingProcessor(embedding_model)
        
    def process_audio_segment(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Process audio segment to extract beginning word features
        
        Args:
            audio_path: Path to audio segment
            
        Returns:
            Dictionary containing various word-based features
        """
        # Convert audio to text
        text = self.speech_processor.audio_to_text(audio_path)
        
        if not text:
            # Return zero features for failed transcription
            return self._get_zero_features()
        
        # Extract beginning words
        beginning_words = self.speech_processor.extract_beginning_words(text, num_words=10)
        
        # Extract sentence starters
        sentence_starters = self.speech_processor.extract_sentence_starters(text, max_sentences=3)
        
        # Detect key phrases
        key_phrases = self.speech_processor.detect_key_phrases(text)
        
        # Generate embeddings
        word_embeddings = self.embedding_processor.embed_words(beginning_words)
        sentence_embeddings = self.embedding_processor.embed_phrase_sequence(sentence_starters)
        fed_sentiment = self.embedding_processor.extract_fed_sentiment_features(text)
        
        # Combine features
        combined_features = torch.cat([
            word_embeddings,
            sentence_embeddings.mean(dim=0),  # Average sentence embeddings
            fed_sentiment
        ])
        
        return {
            'word_embeddings': word_embeddings,
            'sentence_embeddings': sentence_embeddings,
            'fed_sentiment': fed_sentiment,
            'combined_features': combined_features,
            'text': text,
            'beginning_words': beginning_words,
            'key_phrases': key_phrases
        }
    
    def _get_zero_features(self) -> Dict[str, torch.Tensor]:
        """Return zero features for failed processing"""
        embedding_dim = self.embedding_processor.model.config.hidden_size
        fed_features_dim = len(self.embedding_processor.fed_vocabulary) * 2
        
        return {
            'word_embeddings': torch.zeros(embedding_dim),
            'sentence_embeddings': torch.zeros(3, embedding_dim),
            'fed_sentiment': torch.zeros(fed_features_dim),
            'combined_features': torch.zeros(embedding_dim * 2 + fed_features_dim),
            'text': "",
            'beginning_words': [""] * 10,
            'key_phrases': {}
        }
    
    def batch_process_segments(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Process multiple audio segments and return batched features
        
        Args:
            audio_paths: List of audio segment paths
            
        Returns:
            Batched feature tensor [batch_size, feature_dim]
        """
        features = []
        
        for audio_path in audio_paths:
            segment_features = self.process_audio_segment(audio_path)
            features.append(segment_features['combined_features'])
        
        return torch.stack(features) if features else torch.zeros(1, features[0].shape[0])