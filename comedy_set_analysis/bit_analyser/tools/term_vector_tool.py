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

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    vectors_dir: Optional[str] = Field(default=None, description="Directory for bit vectors")
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
        
        # Set vectors directory if not provided
        if self.vectors_dir is None:
            self.vectors_dir = os.path.join(os.path.dirname(self.bits_file), 'bit_vectors')
        
        # Create directories
        os.makedirs(self.vectors_dir, exist_ok=True)
        os.makedirs(self.central_vectors_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize transcript index
        self._transcript_index = None
        self._transcript_cache = {}
        
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
            vector_dirs = [self.vectors_dir, self.central_vectors_dir]
            
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
                backup_dir = os.path.join(os.path.dirname(self.vectors_dir), 'vector_backups')
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
            
    def _build_transcript_index(self, transcript: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float, str]]]:
        """Build an efficient index of transcript segments by show.
        
        Args:
            transcript: List of transcript entries
            
        Returns:
            Dict mapping show_id to list of (start_time, end_time, text) tuples
        """
        try:
            # Group transcript entries by show
            show_segments: Dict[str, List[Tuple[float, float, str]]] = {}
            
            for entry in transcript:
                show_id = entry.get('show_id')
                if not show_id:
                    continue
                    
                start_time = float(entry.get('start_time', 0))
                end_time = float(entry.get('end_time', 0))
                text = entry.get('text', '').strip()
                
                if not text:
                    continue
                
                if show_id not in show_segments:
                    show_segments[show_id] = []
                
                show_segments[show_id].append((start_time, end_time, text))
            
            # Sort segments by start time for each show
            for show_id in show_segments:
                show_segments[show_id].sort(key=lambda x: x[0])
            
            logger.info(f"Built transcript index for {len(show_segments)} shows")
            return show_segments
            
        except Exception as e:
            logger.error(f"Error building transcript index: {e}")
            return {}
    
    def _get_show_segments(self, show_id: str, transcript: List[Dict[str, Any]]) -> List[Tuple[float, float, str]]:
        """Get time-sorted segments for a show with caching."""
        try:
            # Build index if needed
            if self._transcript_index is None:
                self._transcript_index = self._build_transcript_index(transcript)
            
            # Return cached segments
            return self._transcript_index.get(show_id, [])
            
        except Exception as e:
            logger.error(f"Error getting show segments: {e}")
            return []
    
    def extract_bit_text(self, bit_data: Dict[str, Any], transcript: List[Dict[str, Any]]) -> str:
        """Extract bit text from transcript with efficient time-based lookup."""
        try:
            # Get bit info from items array if needed
            if 'items' in bit_data:
                if not bit_data['items']:
                    logger.warning("Empty items array in bit data")
                    return ""
                bit_info = bit_data['items'][0]  # Get first item
            else:
                bit_info = bit_data  # Use as is
            
            # Extract timing info from bit info
            try:
                start_time = float(bit_info.get('start', 0))
                end_time = float(bit_info.get('end', 0))
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid time values in bit data: {e}")
                logger.debug(f"Bit info: {bit_info}")
                return ""
            
            if start_time >= end_time:
                logger.warning(f"Invalid time bounds for bit: {start_time} >= {end_time}")
                return ""
            
            # Check cache first
            cache_key = f"{start_time}_{end_time}"
            if cache_key in self._transcript_cache:
                return self._transcript_cache[cache_key]
            
            # Get relevant segments efficiently
            bit_segments = []
            for entry in transcript:
                if entry.get('type') != 'text':
                    continue
                    
                try:
                    seg_start = float(entry.get('start', 0))
                    seg_end = float(entry.get('end', 0))
                except (TypeError, ValueError):
                    continue
                
                # Skip segments outside our time range
                if seg_end < start_time or seg_start > end_time:
                    continue
                
                text = entry.get('text', '').strip()
                if text:
                    bit_segments.append((seg_start, text))
            
            # Sort segments by start time
            bit_segments.sort(key=lambda x: x[0])
            
            # Join segments
            bit_text = ' '.join(text for _, text in bit_segments).strip()
            
            # Cache result
            self._transcript_cache[cache_key] = bit_text
            
            if not bit_text:
                logger.warning(f"No transcript text found for bit in time range {start_time}-{end_time}")
                logger.debug(f"Bit title: {bit_info.get('title', 'Unknown')}")
                logger.debug(f"Found {len(bit_segments)} segments")
            else:
                logger.info(f"Found {len(bit_segments)} segments for bit: {bit_info.get('title', 'Unknown')}")
            
            return bit_text
            
        except Exception as e:
            logger.error(f"Error extracting bit text: {e}")
            logger.error(f"Bit data: {bit_data}")
            return ""
            
    def process_bit(self, bit_text: str) -> BitVectors:
        """Process a bit of text and return its vector representations."""
        try:
            if not bit_text:
                logger.warning("Empty bit text provided")
                return BitVectors(
                    full_vector=np.zeros(self.dimension, dtype=np.float32),
                    sentence_vectors=np.array([], dtype=np.float32),
                    ngram_vectors=[],
                    punchline_vectors=[]
                )
            
            logger.info("Processing bit text into vectors...")
            
            # Process text with spaCy once
            doc = self.nlp(bit_text)
            
            # Generate sentence vectors first
            sentences = [sent.text.strip() for sent in doc.sents]
            sentence_vectors = self.sentence_model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            sentence_vectors = [round_vector(vec) for vec in sentence_vectors]
            
            # Aggregate sentence vectors for full vector
            full_vector = self._aggregate_sentence_vectors(sentences, sentence_vectors)
            
            # Generate n-gram vectors with intelligent extraction
            scored_ngrams = self._extract_ngrams(doc)
            if scored_ngrams:
                ngram_texts = [text for text, _ in scored_ngrams]
                ngram_scores = [score for _, score in scored_ngrams]
                ngram_embeddings = self.sentence_model.encode(
                    ngram_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                ngram_embeddings = [round_vector(vec) for vec in ngram_embeddings]
                ngram_vectors = list(zip(ngram_texts, ngram_embeddings))
            else:
                ngram_vectors = []
            
            # Get punchlines using semantic analysis
            punchline_vectors = self._get_punchlines(bit_text)
            
            # Create BitVectors object
            vectors = BitVectors(
                full_vector=full_vector,
                sentence_vectors=sentence_vectors,
                ngram_vectors=ngram_vectors,
                punchline_vectors=punchline_vectors
            )
            
            # Log vector information
            logger.info(f"Successfully generated vectors:")
            logger.info(f"- Full vector: {full_vector.shape}")
            logger.info(f"- Sentences: {len(sentence_vectors)}")
            logger.info(f"- N-grams: {len(ngram_vectors)}")
            logger.info(f"- Punchlines: {len(punchline_vectors)}")
            
            return vectors
            
        except Exception as e:
            logger.error(f"Error processing bit text: {e}")
            raise
            
    def read_metadata(self) -> Dict[str, Any]:
        """Read and return the metadata."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
            
    def create_show_identifier(self, metadata: Dict[str, Any]) -> str:
        """Create a unique show identifier from metadata."""
        date = datetime.strptime(metadata['date_of_show'], "%d %b %Y, %H:%M")
        show_name = metadata['name_of_show'].replace(" ", "_")
        return f"{date.strftime('%Y%m%d')}_{show_name}"
        
    def combine_bit_data(self, bit: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Combine bit and metadata into standard format."""
        return {
            'show_info': {
                'show_identifier': self.create_show_identifier(metadata),
                'comedian': metadata['comedian'],
                'name_of_show': metadata['name_of_show'],
                'date_of_show': metadata['date_of_show'],
                'name_of_venue': metadata['name_of_venue'],
                'length_of_set': metadata['length_of_set'],
                'laughs_per_minute': metadata['laughs_per_minute']
            },
            'bit_info': {
                'title': bit['title'],
                'start': bit['start'],
                'end': bit['end'],
                'joke_types': bit.get('joke_types', []),
                'themes': bit.get('themes', []),
                'lpm': bit.get('lpm', 0),
                'term_vector_hash': bit.get('term_vector_hash', '')
            }
        }
        
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

    def _extract_ngrams(self, doc: spacy.tokens.Doc) -> List[Tuple[str, float]]:
        """Extract n-grams from text with intelligent filtering and scoring."""
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
                        
                        sent_spans.append(span)
                
                # Filter overlapping spans
                filtered_spans = spacy.util.filter_spans(sent_spans)
                spans.extend(filtered_spans)
            
            # Score and filter n-grams
            scored_ngrams = []
            for span in spans:
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
                    scored_ngrams.append((span.text.strip(), score))
            
            # Sort by score and deduplicate
            scored_ngrams.sort(key=lambda x: x[1], reverse=True)
            
            # Remove near-duplicates (substrings)
            final_ngrams = []
            seen_texts = set()
            for text, score in scored_ngrams:
                # Skip if this is a substring of an already seen n-gram
                if any(text in seen for seen in seen_texts):
                    continue
                # Skip if this contains an already seen n-gram
                if any(seen in text for seen in seen_texts):
                    continue
                final_ngrams.append((text, score))
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
            
    def save_bit_vectors(self, bit_id: str, vectors: BitVectors) -> None:
        """Save bit vectors to both local and central directories."""
        try:
            # Save to local directory
            local_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            vectors.save(local_file)
            logger.info(f"Saved vectors to {local_file}")
            
            # Save to central directory
            central_file = os.path.join(self.central_vectors_dir, f"{bit_id}.npz")
            vectors.save(central_file)
            logger.info(f"Saved vectors to {central_file}")
            
        except Exception as e:
            logger.error(f"Error saving vectors for {bit_id}: {e}")
            raise
            
    def load_bit_vectors(self, bit_id: str) -> Optional[BitVectors]:
        """
        Load bit vectors from file.
        
        Args:
            bit_id: Unique identifier for the bit
            
        Returns:
            BitVectors object if found, None otherwise
        """
        vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
        
        try:
            if not os.path.exists(vector_file):
                logger.warning(f"No vector file found for bit: {bit_id}")
                return None
                
            # Load compressed vectors
            with np.load(vector_file) as data:
                # Round loaded vectors to maintain consistency
                full_vector = np.round(data['full_vector'], VECTOR_PRECISION)
                sentence_vectors = [np.round(v, VECTOR_PRECISION) for v in data['sentence_vectors']]
                
                # Reconstruct n-gram vectors
                ngram_vectors = list(zip(
                    data['ngram_texts'],
                    [np.round(v, VECTOR_PRECISION) for v in data['ngram_vectors']]
                ))
                
                # Reconstruct punchline vectors
                punchline_vectors = list(zip(
                    data['punchline_texts'],
                    [np.round(v, VECTOR_PRECISION) for v in data['punchline_vectors']],
                    data['punchline_weights']
                ))
                
                return BitVectors(
                    full_vector=full_vector,
                    sentence_vectors=sentence_vectors,
                    ngram_vectors=ngram_vectors,
                    punchline_vectors=punchline_vectors
                )
                
        except Exception as e:
            logger.error(f"Error loading vectors for bit {bit_id}: {e}")
            return None
        
    def add_to_database(self, bit_id: str, bit_data: Dict[str, Any], vectors: BitVectors) -> None:
        """Add bit to central database."""
        # Get central database path
        registry_file = os.path.join(self.central_vectors_dir, "bit_registry.json")
        
        # Save vectors to central database
        vector_file = os.path.join(self.central_vectors_dir, f"{bit_id}.npz")
        self.save_bit_vectors(bit_id, vectors)
        
        # Update registry
        registry = {}
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                registry = json.load(f)
                
        registry[bit_id] = {
            'title': bit_data['bit_info']['title'],
            'show': bit_data['show_info']['name_of_show'],
            'date': bit_data['show_info']['date_of_show'],
            'comedian': bit_data['show_info']['comedian'],
            'venue': bit_data['show_info'].get('name_of_venue', ''),
            'vector_file': vector_file
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
        logger.info(f"Added bit to central database: {bit_id}")
        
    def process_bits(self) -> None:
        """Process all bits in the input directory."""
        logger.info(f"Processing bits from: {self.bits_file}")
        
        # Read input files
        with open(self.bits_file, 'r') as f:
            bits_schema = json.load(f)
            
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
            
        with open(self.transcript_file, 'r') as f:
            transcript = json.load(f)
            
        # Get bits from schema
        bits = bits_schema.get('items', [])
        if not bits:
            logger.error("No bits found in bits.json")
            return
            
        # Process each bit
        for bit in bits:
            if not isinstance(bit, dict):
                logger.error(f"Invalid bit format: {bit}")
                continue
                
            try:
                logger.info(f"\nProcessing bit: {bit.get('title', 'Untitled')}")
                
                # Create unique ID for this bit
                bit_id = f"{self.create_show_identifier(metadata)}_{bit['title'].lower().replace(' ', '_')}"
                
                # Skip if vectors exist and not regenerating
                vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
                if os.path.exists(vector_file) and not self.regenerate:
                    logger.info(f"Vectors already exist for bit: {bit['title']}")
                    continue
                
                # Extract bit text from transcript
                bit_text = self.extract_bit_text(bit, transcript)
                if not bit_text:
                    logger.warning(f"No text found for bit: {bit['title']}")
                    continue
                
                # Generate vectors
                bit_vectors = self.process_bit(bit_text)
                
                # Save vectors
                self.save_bit_vectors(bit_id, bit_vectors)
                
                # Add to central database
                bit_data = self.combine_bit_data(bit, metadata)
                self.add_to_database(bit_id, bit_data, bit_vectors)
                
            except Exception as e:
                logger.error(f"Error processing bit {bit.get('title', 'Untitled')}: {str(e)}")
                continue
                
    def run(self) -> None:
        """Run the vector generation tool."""
        logger.info(f"Processing bits from: {self.bits_file}")
        self.process_bits()
        logger.info("Finished processing bits")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate term vector representations for comedy bits.'
    )
    parser.add_argument('-b', '--bits-file', type=str, required=True,
                      help='Path to bits JSON file')
    parser.add_argument('-m', '--metadata-file', type=str, required=True,
                      help='Path to metadata JSON file')
    parser.add_argument('-t', '--transcript-file', type=str, required=True,
                      help='Path to transcript_clean.json file')
    parser.add_argument('-v', '--vectors-dir', type=str,
                      help='Directory to store vector files (default: bit_vectors/ in bits directory)')
    parser.add_argument('-r', '--regenerate', action='store_true',
                      help='Force regeneration of all vectors')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(message)s')
    
    tool = TermVectorTool(
        bits_file=args.bits_file,
        metadata_file=args.metadata_file,
        transcript_file=args.transcript_file,
        vectors_dir=args.vectors_dir,
        regenerate=args.regenerate
    )
    
    # Run the tool directly without threading
    tool.run()
