#!/usr/bin/env python3

"""
Term vector generation tool for comedy bits using sentence transformers and n-grams.
Implements a multi-level embedding approach:
1. Sentence-level embeddings using SBERT
2. N-gram extraction and weighting
3. Punchline detection and emphasis
"""

import os
import sys

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, ClassVar, Set
from datetime import datetime
import spacy
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from base_tool import BaseTool
from bit_vectors import BitVectors
from bit_utils import select_bit, flatten_bit
import threading

logger = logging.getLogger(__name__)

VECTOR_PRECISION = 6  # Number of decimal places for vector rounding

def round_vector(vector: np.ndarray) -> np.ndarray:
    """Round vector to consistent precision."""
    return np.round(vector, VECTOR_PRECISION)

class TermVectorTool(BaseModel):
    """Tool for processing bits and generating vectors."""
    
    # Class-level lock for thread-safe initialization
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    # Class-level model instances (singleton pattern)
    _sentence_model: ClassVar[Optional[SentenceTransformer]] = None
    _nlp: ClassVar[Optional[spacy.language.Language]] = None
    _dimension: ClassVar[Optional[int]] = None
    _initialized: ClassVar[bool] = False
    _stopwords: ClassVar[Optional[Set[str]]] = None
    
    # Input files
    bits_file: str = Field(description="Path to bits.json")
    metadata_file: str = Field(description="Path to metadata.json")
    transcript_file: str = Field(description="Path to transcript_clean.json")
    
    # Directories
    central_vectors_dir: str = Field(default=os.path.expanduser("~/.comedybot/vectors"))
    
    # Configuration
    regenerate: bool = Field(default=False, description="Whether to regenerate existing vectors")
    
    # Instance-level model references
    sentence_model: Optional[Any] = Field(default=None, description="Sentence transformer model")
    nlp: Optional[Any] = Field(default=None, description="Spacy model")
    dimension: Optional[int] = Field(default=None, description="Vector dimension")
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        super().__init__(**data)
        
        # Create directories
        os.makedirs(self.central_vectors_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize transcript index
        self._transcript_index = None
        
    @classmethod
    def _load_models(cls):
        """Load models once at class level with proper synchronization."""
        # Fast check without lock
        if cls._initialized:
            return
            
        # Acquire lock for initialization
        with cls._lock:
            # Check again with lock
            if cls._initialized:
                return
                
            try:
                # Disable progress bar logging
                logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
                
                # Load sentence transformer
                cls._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                cls._dimension = cls._sentence_model.get_sentence_embedding_dimension()
                
                # Load spaCy with optimized settings
                cls._nlp = spacy.load("en_core_web_md", 
                                    disable=['ner', 'textcat', 'parser'],
                                    enable=['sentencizer'])
                cls._nlp.add_pipe('sentencizer')
                
                # Initialize stopwords
                cls._stopwords = set(cls._nlp.Defaults.stop_words)
                
                # Mark as initialized
                cls._initialized = True
                
                logger.info(f"Loaded models with dimension: {cls._dimension}")
                
            except ModuleNotFoundError as e:
                logger.error(f"Required model not found: {e}")
                raise
            except OSError as e:
                logger.error(f"Error loading model files: {e}")
                raise
        
    def _init_models(self):
        """Initialize model references with proper error handling."""
        try:
            # Load models if not already loaded
            self._load_models()
            
            # Set instance references to class models
            self.sentence_model = self._sentence_model
            self.nlp = self._nlp
            self.dimension = self._dimension
            
            if not all([self.sentence_model, self.nlp, self.dimension]):
                raise RuntimeError("Models failed to initialize properly")
            
            # Validate existing vectors
            self._validate_existing_vectors()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
            
    def _validate_existing_vectors(self) -> None:
        """Validate dimensions of existing vectors and log issues."""
        try:
            mismatched_vectors = []
            corrupt_vectors = []
            
            # Check both local and central vector directories
            vector_dirs = [self.central_vectors_dir]
            
            for vector_dir in vector_dirs:
                if not os.path.exists(vector_dir):
                    continue
                    
                for file in os.listdir(vector_dir):
                    if not file.endswith('.npz'):
                        continue
                        
                    vector_file = os.path.join(vector_dir, file)
                    try:
                        with np.load(vector_file) as data:
                            if 'full_vector' not in data:
                                logger.warning(f"Missing full_vector in {file}")
                                corrupt_vectors.append(vector_file)
                                continue
                                
                            vec_dim = data['full_vector'].shape[0]
                            if vec_dim != self.dimension:
                                logger.warning(
                                    f"Dimension mismatch in {file}: "
                                    f"expected {self.dimension}, found {vec_dim}"
                                )
                                mismatched_vectors.append((vector_file, vec_dim))
                                
                            # Validate other vector components
                            for key in ['sentence_vectors', 'ngram_vectors', 'punchline_vectors']:
                                if key in data and len(data[key]) > 0:
                                    component_dim = data[key][0].shape[0]
                                    if component_dim != self.dimension:
                                        logger.warning(
                                            f"Dimension mismatch in {file} {key}: "
                                            f"expected {self.dimension}, found {component_dim}"
                                        )
                                        mismatched_vectors.append((vector_file, component_dim))
                                        
                    except (np.lib.format.FormatError, OSError, KeyError, IndexError) as e:
                        logger.error(f"Error reading vector file {file}: {e}")
                        corrupt_vectors.append(vector_file)
            
            # Log summary of issues
            if mismatched_vectors:
                logger.warning(
                    f"Found {len(mismatched_vectors)} vectors with dimension mismatches:\n" +
                    "\n".join(f"- {file}: dimension {dim}" for file, dim in mismatched_vectors)
                )
                
            if corrupt_vectors:
                logger.warning(
                    f"Found {len(corrupt_vectors)} corrupt or invalid vector files:\n" +
                    "\n".join(f"- {file}" for file in corrupt_vectors)
                )
                
            # If regenerate is True and there are issues, clear problematic vectors
            if self.regenerate and (mismatched_vectors or corrupt_vectors):
                logger.info("Regenerate flag is set - clearing problematic vectors")
                
                # Move problematic files to backup directory
                backup_dir = os.path.join(os.path.dirname(self.central_vectors_dir), 'vector_backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                for file, _ in mismatched_vectors:
                    backup_file = os.path.join(
                        backup_dir,
                        f"{os.path.basename(file)}.{timestamp}.dimension_mismatch"
                    )
                    os.rename(file, backup_file)
                    logger.info(f"Backed up mismatched vector file: {file} -> {backup_file}")
                
                for file in corrupt_vectors:
                    backup_file = os.path.join(
                        backup_dir,
                        f"{os.path.basename(file)}.{timestamp}.corrupt"
                    )
                    os.rename(file, backup_file)
                    logger.info(f"Backed up corrupt vector file: {file} -> {backup_file}")
            
            elif mismatched_vectors or corrupt_vectors:
                logger.warning(
                    "Found vector issues but regenerate=False. "
                    "Set regenerate=True to automatically handle problematic vectors."
                )
            
        except Exception as e:
            logger.error(f"Error validating vectors: {e}")
            raise
        
    def _get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a list of texts."""
        try:
            if not texts:
                return np.array([])
            
            # Encode all texts in a single batch with progress bar disabled
            embeddings = self.sentence_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings = [round_vector(vec) for vec in embeddings]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([])
            
    def _aggregate_sentence_vectors(self, sentences: List[str], vectors: np.ndarray) -> np.ndarray:
        """Aggregate sentence vectors into a full text vector using weighted averaging."""
        try:
            if len(sentences) == 0:
                return np.zeros(self.dimension)
            if len(sentences) == 1:
                return vectors[0]
            
            # Calculate weights based on sentence properties
            weights = np.ones(len(sentences))
            
            for i, sent in enumerate(sentences):
                # 1. Length factor (longer sentences often more important)
                words = sent.split()
                length_factor = min(len(words) / 10, 1.5)  # Cap at 1.5x
                weights[i] *= length_factor
                
                # 2. Position factor (first/last sentences often more important)
                if i == 0 or i == len(sentences) - 1:
                    weights[i] *= 1.2
                
                # 3. Question/exclamation factor (often key points)
                if sent.strip().endswith(('?', '!')):
                    weights[i] *= 1.1
                
                # 4. Content word ratio
                words = set(sent.lower().split())
                if not self._stopwords:  # Fallback if stopwords not loaded
                    content_ratio = 1.0
                else:
                    content_words = words - self._stopwords
                    if words:  # Avoid division by zero
                        content_ratio = len(content_words) / len(words)
                    else:
                        content_ratio = 1.0
                weights[i] *= (0.5 + content_ratio)  # Scale from 0.5 to 1.5
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Weighted average of sentence vectors
            weighted_vectors = vectors * weights[:, np.newaxis]
            aggregated_vector = np.sum(weighted_vectors, axis=0)
            
            # Normalize the final vector (use L2 norm)
            norm = np.linalg.norm(aggregated_vector)
            if norm > 0:
                aggregated_vector = aggregated_vector / norm
            
            # Ensure vector is contiguous and in the right format for FAISS
            return np.ascontiguousarray(aggregated_vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error aggregating sentence vectors: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def _extract_ngrams(self, doc: spacy.tokens.Doc) -> List[Tuple[str, float, int]]:
        """Extract n-grams from text with intelligent filtering and scoring.
        
        Returns:
            List of tuples containing (text, score, char_position)
        """
        try:
            # Initialize n-gram spans
            spans = []
            min_length = 3  # Minimum words in n-gram
            max_length = 8  # Maximum words in n-gram
            
            # Extract candidate spans from each sentence
            for sent in doc.sents:
                # Get all possible spans within length bounds
                sent_spans = []
                for start in range(len(sent)):
                    for length in range(min_length, min(max_length + 1, len(sent) - start + 1)):
                        span = sent[start:start + length]
                        
                        # Basic filtering
                        if any(t.is_punct for t in span):
                            continue
                            
                        # Skip if span starts/ends with stopwords or determiners
                        if span[0].is_stop or span[-1].is_stop:
                            continue
                            
                        # Skip if span has no content words
                        if not any(not t.is_stop for t in span):
                            continue
                        
                        sent_spans.append((span, span[0].idx))  # Store character position
                
                # Filter overlapping spans
                filtered_spans = [(span, pos) for span, pos in sent_spans]
                spans.extend(filtered_spans)
            
            # Score and filter n-grams
            scored_ngrams = []
            for span, char_pos in spans:
                # Calculate importance score based on multiple factors
                score = 1.0
                
                # 1. Length factor (prefer medium length)
                length = len(span)
                length_score = 1.0 - abs(length - 5) / 5  # Peak at length 5
                score *= length_score
                
                # 2. Content word ratio
                content_words = sum(1 for t in span if not t.is_stop)
                content_ratio = content_words / length
                score *= content_ratio
                
                # 3. Noun phrase bonus
                if span.root.pos_ in ['NOUN', 'PROPN']:
                    score *= 1.2
                
                # 4. Verb phrase bonus
                if span.root.pos_ == 'VERB':
                    score *= 1.1
                
                # Add if score is high enough
                if score > 0.4:  # Threshold for keeping n-grams
                    scored_ngrams.append((span.text.strip(), score, char_pos))
            
            # Sort by score and deduplicate
            scored_ngrams.sort(key=lambda x: x[1], reverse=True)
            
            # Remove near-duplicates (substrings)
            final_ngrams = []
            seen_texts = set()
            for text, score, pos in scored_ngrams:
                # Skip if this is a substring of an already seen n-gram
                if any(text in seen for seen in seen_texts):
                    continue
                # Skip if this contains an already seen n-gram
                if any(seen in text for seen in seen_texts):
                    continue
                final_ngrams.append((text, score, pos))
                seen_texts.add(text)
            
            return final_ngrams
            
        except Exception as e:
            logger.error(f"Error extracting n-grams: {e}")
            return []

    def _get_punchlines(self, text: str) -> List[Tuple[str, np.ndarray, float]]:
        """Extract potential punchlines using semantic analysis."""
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if not sentences:
                return []
            
            # Get embeddings for all sentences
            sentence_texts = [sent.text.strip() for sent in sentences]
            sentence_vecs = self._get_embeddings(sentence_texts)
            sentence_vecs = np.round(sentence_vecs, VECTOR_PRECISION)
            
            punchlines = []
            window_size = 3  # Context window size
            
            # Find potential punchlines using semantic and structural analysis
            for i, (sent, vec) in enumerate(zip(sentences, sentence_vecs)):
                score = 0.0
                is_punchline = False
                
                # 1. Semantic shift analysis
                if i > 0:
                    # Look at previous context window
                    start_idx = max(0, i - window_size)
                    prev_vecs = sentence_vecs[start_idx:i]
                    context_vec = np.mean(prev_vecs, axis=0)
                    
                    # Calculate semantic shift
                    semantic_shift = 1 - np.dot(context_vec, vec)
                    if semantic_shift > 0.3:  # Significant semantic change
                        score += semantic_shift
                        is_punchline = True
                
                # 2. Position analysis
                if i == len(sentences) - 1:  # Last sentence of paragraph
                    score += 0.3
                    is_punchline = True
                
                # 3. Length analysis
                rel_length = len(sent.text.split()) / np.mean([len(s.text.split()) for s in sentences])
                if rel_length < 0.7:  # Shorter than average (setup-punchline pattern)
                    score += 0.2
                    is_punchline = True
                
                # Add if identified as punchline
                if is_punchline:
                    score = min(1.0, score)  # Normalize score
                    punchlines.append((sent.text.strip(), vec, score))
            
            # Sort by score and return top punchlines
            punchlines.sort(key=lambda x: x[2], reverse=True)
            return punchlines[:3]  # Return top 3 punchlines
            
        except Exception as e:
            logger.error(f"Error extracting punchlines: {e}")
            return []

    def process_bit(self, bit_text: str) -> BitVectors:
        """Process a bit of text and return its vector representations."""
        try:
            # Process text with spaCy
            doc = self.nlp(bit_text)
            
            # Get sentence embeddings
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_embeddings = self._get_embeddings(sentences)
            
            # Get full text vector through weighted aggregation
            full_vector = self._aggregate_sentence_vectors(sentences, sentence_embeddings)
            
            # Generate n-gram vectors with intelligent extraction
            scored_ngrams = self._extract_ngrams(doc)
            if scored_ngrams:
                ngram_texts = [text for text, _, _ in scored_ngrams]
                ngram_scores = [score for _, score, _ in scored_ngrams]
                ngram_positions = [pos for _, _, pos in scored_ngrams]
                ngram_embeddings = self.sentence_model.encode(
                    ngram_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                ngram_embeddings = [round_vector(vec) for vec in ngram_embeddings]
                ngram_vectors = list(zip(ngram_texts, ngram_embeddings, ngram_positions))
            else:
                ngram_vectors = []
            
            # Get punchline vectors
            punchlines = self._get_punchlines(bit_text)
            if punchlines:
                punchline_texts = [text for text, vec, weight in punchlines]
                punchline_weights = [weight for text, vec, weight in punchlines]
                punchline_embeddings = [vec for text, vec, weight in punchlines]
                punchline_vectors = list(zip(punchline_texts, punchline_embeddings, punchline_weights))
            else:
                punchline_vectors = []
            
            # Create BitVectors object
            vectors = BitVectors(
                full_vector=full_vector,
                sentence_vectors=sentence_embeddings,
                ngram_vectors=ngram_vectors,
                punchline_vectors=punchline_vectors
            )
            
            return vectors
            
        except Exception as e:
            import traceback
            
            # Get the full traceback
            tb = traceback.format_exc()
            
            # Log detailed error information
            logger.error("=============== Error processing bit ===============")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"\nFull traceback:\n{tb}")
            
            # Log processing pipeline state
            logger.error("\n=============== Processing state ===============")
            steps_completed = []
            if 'doc' in locals():
                steps_completed.append("Text processed with spaCy")
            if 'sentences' in locals():
                steps_completed.append(f"Sentences extracted (count: {len(sentences)})")
            if 'sentence_embeddings' in locals():
                steps_completed.append("Sentence embeddings generated")
            if 'full_vector' in locals():
                steps_completed.append("Full text vector aggregated")
            if 'scored_ngrams' in locals():
                steps_completed.append(f"N-grams extracted (count: {len(scored_ngrams)})")
            if 'ngram_embeddings' in locals():
                steps_completed.append("N-gram embeddings generated")
            if 'punchlines' in locals():
                steps_completed.append(f"Punchlines extracted (count: {len(punchlines)})")
            
            logger.error("Processing pipeline progress:")
            for i, step in enumerate(steps_completed, 1):
                logger.error(f"{i}. {step}")
            
            # Log data structures at point of failure
            logger.error("\n=============== Data structures ===============")
            if 'scored_ngrams' in locals() and scored_ngrams:
                logger.error(f"\nN-gram format example:")
                logger.error(f"First n-gram: {scored_ngrams[0]}")
                logger.error(f"N-gram tuple length: {len(scored_ngrams[0])}")
                logger.error(f"N-gram types: {[type(x).__name__ for x in scored_ngrams[0]]}")
            
            if 'punchlines' in locals() and punchlines:
                logger.error(f"\nPunchline format example:")
                logger.error(f"First punchline: {punchlines[0]}")
                logger.error(f"Punchline tuple length: {len(punchlines[0])}")
                logger.error(f"Punchline types: {[type(x).__name__ for x in punchlines[0]]}")
            
            logger.error("\n===============================================")
            
            raise
                
    def run(self) -> None:
        """Pass for now"""
        logger.info("Term vector tool running...")
        return
