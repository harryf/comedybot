#!/usr/bin/env python3

"""
Bit vector database using multi-level comparison:
1. Full bit vector comparison
2. Sentence-level matching
3. N-gram and punchline detection
"""

import os
import sys

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import numpy as np
import faiss
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from bit_vectors import BitVectors

# Configure logging
logger = logging.getLogger(__name__)

def round_vector(vector: np.ndarray) -> np.ndarray:
    """Round vector to consistent precision."""
    return np.round(vector, BitVectorDBConfig.VECTOR_PRECISION)

class BitVectorDBConfig:
    """Configuration constants for BitVectorDB."""
    
    # Vector precision
    VECTOR_PRECISION = 6  # Number of decimal places for vector rounding
    
    # Match thresholds
    HARD_MATCH_THRESHOLD = 0.7
    SOFT_MATCH_THRESHOLD = 0.6
    
    # Component weights for scoring - keys match usage in find_matching_bits
    WEIGHTS = {
        'full': 0.45,  # Weight for full vector comparison
        'sent': 0.30,  # Weight for sentence matches
        'ngram': 0.15, # Weight for n-gram matches
        'punch': 0.10  # Weight for punchline matches
    }
    
    # Thresholds for different components
    THRESHOLDS = {
        'full_vector': 1.2,   # Keep high to avoid false full matches
        'sentences': 0.8,     # Lower to catch subset matches better
        'ngrams': 1.2,        # Increased from 1.0 to reduce false n-gram matches
        'punchlines': 1.2     # Keep same
    }
    
    # Boost factors for strong component matches
    BOOST_FACTORS = {
        'full': 1.3,    # Boost for strong full vector matches
        'sent': 1.2,    # Boost for strong sentence matches
        'ngram': 1.1,   # Boost for strong n-gram matches
        'punch': 1.2    # Boost for strong punchline matches
    }
    
    # Search configuration
    SEARCH = {
        'initial_candidates': 15,  # Number of initial candidates to consider
        'sentence_candidates': 5,  # Number of sentence matches to consider per sentence
        'ngram_candidates': 3,     # Number of n-gram matches to consider per n-gram
        'punchline_candidates': 3  # Number of punchline matches to consider per punchline
    }
    
    # Component-specific thresholds
    COMPONENT_THRESHOLDS = {
        'sentence': {
            'early': 0.9,  # Threshold for early sentences
            'late': 1.1    # Threshold for later sentences
        },
        'ngram': 0.9,      # Threshold for n-gram matches
        'punchline': 1.1   # Threshold for punchline matches
    }
    
    # Score boost thresholds
    BOOST_THRESHOLDS = {
        'full_vector': 0.75,
        'sentences': 0.7,
        'ngrams': 0.6,
        'punchlines': 0.7,
        'multi_component': 0.6  # Threshold for considering a component "strong"
    }
    
    # Multi-component boost factor
    MULTI_COMPONENT_BOOST = 1.2  # Boost factor when multiple components are strong


class BitStorageManager:
    """Manages filesystem operations for bit vector storage."""

    def __init__(self, base_dir: str = '~/.comedybot/'):
        """Initialize storage manager with base directory."""
        self.base_dir = os.path.expanduser(base_dir)
        self.vectors_dir = os.path.join(self.base_dir, 'vectors/')
        self.indices_dir = os.path.join(self.base_dir, 'indices/')
        self.registry_file = os.path.join(self.base_dir, 'bit_registry.json')
        self._init_directories()

    def _init_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.base_dir, self.vectors_dir, self.indices_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied creating directory {directory}: {e}")
                raise
            except OSError as e:
                logger.error(f"OS error creating directory {directory}: {e}")
                raise

    def load_registry(self) -> Dict[str, Any]:
        """Load bit registry from file."""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            return {}
        except FileNotFoundError:
            logger.error(f"Registry file not found: {self.registry_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing registry file: {e}")
            return {}
        except PermissionError:
            logger.error(f"Permission denied accessing registry file: {self.registry_file}")
            return {}

    def save_registry(self, registry: Dict[str, Any]):
        """Save bit registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def save_bit_vectors(self, bit_id: str, vectors: BitVectors):
        """Save bit vectors to file."""
        try:
            vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            np.savez(
                vector_file,
                full_vector=round_vector(vectors.full_vector),
                sentence_vectors=np.array([round_vector(vec) for vec in vectors.sentence_vectors]) if vectors.sentence_vectors else np.array([]),
                ngram_vectors=np.array([round_vector(vec) for _, vec in vectors.ngram_vectors]) if vectors.ngram_vectors else np.array([]),
                punchline_vectors=np.array([round_vector(vec) for _, vec, _ in vectors.punchline_vectors]) if vectors.punchline_vectors else np.array([])
            )
            logger.info(f"Saved vectors for bit {bit_id}")
            return vector_file
        except (OSError, ValueError) as e:
            logger.error(f"Error saving vectors for bit {bit_id}: {e}")
            raise

    def save_indices(self, indices: Dict[str, Any]):
        """Save FAISS indices to disk."""
        try:
            for name, index in indices.items():
                if index is not None:
                    faiss.write_index(index, os.path.join(self.indices_dir, f'{name}_index.faiss'))
            logger.info("Saved FAISS indices to disk")
        except (OSError, RuntimeError) as e:
            logger.error(f"Error saving indices: {e}")
            raise

    def load_bit_vectors(self, bit_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load vectors for a specific bit."""
        try:
            vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            if os.path.exists(vector_file):
                return dict(np.load(vector_file))
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading vectors for bit {bit_id}: {e}")
            return None

    def list_vector_files(self) -> List[str]:
        """List all vector files in the vectors directory."""
        return [f for f in os.listdir(self.vectors_dir) if f.endswith(".npz")]

    def delete_bit_data(self, bit_id: str):
        """Delete all data associated with a bit."""
        try:
            vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            if os.path.exists(vector_file):
                os.remove(vector_file)
        except OSError as e:
            logger.error(f"Error deleting bit data for {bit_id}: {e}")
            raise

class BitMatch(BaseModel):
    """
    Represents a matching bit from the database.
    """
    bit_id: str
    title: str
    overall_score: float
    sentence_matches: List[Tuple[str, str, float]] = Field(default_factory=list)
    ngram_matches: List[Tuple[str, str, float]] = Field(default_factory=list)
    punchline_matches: List[Tuple[str, str, float]] = Field(default_factory=list)
    show_info: Optional[Dict[str, Any]] = None
    match_type: str = Field(default="exact")  # "exact", "soft", or "none"

class BitVectorDB:
    """Database for Comedy Bits with FAISS indexing."""

    def __init__(self, dimension: int = 384, similarity_threshold: float = 0.7):
        """Initialize the bit database."""
        self.full_index = None
        self.sentence_index = None
        self.ngram_index = None
        self.punchline_index = None
        
        # Initialize mapping dictionaries
        self.sentence_map = {}
        self.ngram_map = {}
        self.punchline_map = {}
        
        try:
            self.dimension = dimension
            self.similarity_threshold = similarity_threshold

            # Set FAISS to single thread for deterministic results
            faiss.omp_set_num_threads(1)

            # Initialize storage manager
            self.storage = BitStorageManager()

            # Load or initialize registry
            self.registry = self.storage.load_registry()

            # Initialize FAISS indices
            self._init_indices()

            logger.info(f"Initialized BitDB with dimension: {dimension}, threshold: {similarity_threshold}")

        except RuntimeError as e:
            logger.error(f"FAISS initialization error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid parameter value: {e}")
            raise

    def _load_registry(self) -> Dict[str, Any]:
        """Load bit registry from file."""
        return self.storage.load_registry()

    def _save_registry(self):
        """Save bit registry to file."""
        self.storage.save_registry(self.registry)

    def _save_bit_vectors(self, bit_id: str, vectors: BitVectors):
        """Save bit vectors to file."""
        return self.storage.save_bit_vectors(bit_id, vectors)

    def _save_indices(self):
        """Save FAISS indices to disk."""
        self.storage.save_indices({
            'full': self.full_index,
            'sentence': self.sentence_index,
            'ngram': self.ngram_index,
            'punchline': self.punchline_index
        })

    def _add_bit_vectors_to_indices(self, bit_id: str, vectors: BitVectors):
        """Add a bit's vectors to the indices."""
        try:
            # Normalize and reshape vectors
            full_vector = round_vector(vectors.full_vector).reshape(1, -1)
            faiss.normalize_L2(full_vector)

            # Add to indices
            if self.full_index is not None:
                self.full_index.add(full_vector)

            # Add sentence vectors if available
            if vectors.sentence_vectors:
                sentence_vectors = np.array([round_vector(vec) for vec in vectors.sentence_vectors])
                sentence_vectors = sentence_vectors.reshape(len(sentence_vectors), -1)
                faiss.normalize_L2(sentence_vectors)
                if self.sentence_index is not None:
                    current_idx = self.sentence_index.ntotal
                    self.sentence_index.add(sentence_vectors)
                    # Update sentence map
                    for i, sent_vec in enumerate(vectors.sentence_vectors):
                        self.sentence_map[current_idx + i] = (bit_id, str(sent_vec))

            # Add ngram vectors if available
            if vectors.ngram_vectors:
                ngram_vectors = np.array([round_vector(vec) for _, vec in vectors.ngram_vectors])
                ngram_vectors = ngram_vectors.reshape(len(ngram_vectors), -1)
                faiss.normalize_L2(ngram_vectors)
                if self.ngram_index is not None:
                    current_idx = self.ngram_index.ntotal
                    self.ngram_index.add(ngram_vectors)
                    # Update ngram map
                    for i, (ng_text, _) in enumerate(vectors.ngram_vectors):
                        self.ngram_map[current_idx + i] = (bit_id, ng_text)

            # Add punchline vectors if available
            if vectors.punchline_vectors:
                punchline_vectors = np.array([round_vector(vec) for _, vec, _ in vectors.punchline_vectors])
                punchline_vectors = punchline_vectors.reshape(len(punchline_vectors), -1)
                faiss.normalize_L2(punchline_vectors)
                if self.punchline_index is not None:
                    current_idx = self.punchline_index.ntotal
                    self.punchline_index.add(punchline_vectors)
                    # Update punchline map
                    for i, (p_text, _, _) in enumerate(vectors.punchline_vectors):
                        self.punchline_map[current_idx + i] = (bit_id, p_text)

            # Save vector file and update registry
            vector_file = self.storage.save_bit_vectors(bit_id, vectors)
            if bit_id not in self.registry:
                self.registry[bit_id] = {
                    'id': bit_id,
                    'vector_file': vector_file,
                    'added_at': datetime.now().isoformat()
                }
                self.storage.save_registry(self.registry)

            # Save indices
            self.storage.save_indices({
                'full': self.full_index,
                'sentence': self.sentence_index,
                'ngram': self.ngram_index,
                'punchline': self.punchline_index
            })

            logger.info(f"Successfully added vectors for bit: {bit_id}")

        except Exception as e:
            logger.error(f"Error adding bit vectors: {e}")
            raise

    def _init_indices(self):
        """Initialize FAISS indices."""
        try:
            # Initialize indices using cosine similarity (L2 normalized + dot product)
            self.full_index = faiss.IndexFlatL2(self.dimension)
            self.sentence_index = faiss.IndexFlatL2(self.dimension)
            self.ngram_index = faiss.IndexFlatL2(self.dimension)
            self.punchline_index = faiss.IndexFlatL2(self.dimension)
            
            # Reset mapping dictionaries
            self.sentence_map = {}
            self.ngram_map = {}
            self.punchline_map = {}

            # Load existing vectors in batches
            batch_size = 1000
            vectors_batch = []
            ids_batch = []

            for bit_id in self.registry:
                try:
                    vector_file = os.path.join(self.storage.vectors_dir, f"{bit_id}.npz")
                    if os.path.exists(vector_file):
                        with np.load(vector_file) as data:
                            # Add to full vector batch
                            vectors_batch.append(round_vector(data['full_vector']))
                            ids_batch.append(bit_id)
                            
                            # Add to sentence index
                            if 'sentence_vectors' in data and len(data['sentence_vectors']) > 0:
                                sentence_vectors = data['sentence_vectors']
                                current_idx = self.sentence_index.ntotal
                                self.sentence_index.add(sentence_vectors)
                                for i, sent_vec in enumerate(sentence_vectors):
                                    self.sentence_map[current_idx + i] = (bit_id, str(sent_vec))
                            
                            # Add to ngram index
                            if 'ngram_vectors' in data and len(data['ngram_vectors']) > 0:
                                ngram_vectors = data['ngram_vectors']
                                current_idx = self.ngram_index.ntotal
                                self.ngram_index.add(ngram_vectors)
                                for i in range(len(ngram_vectors)):
                                    self.ngram_map[current_idx + i] = (bit_id, f"ngram_{i}")
                            
                            # Add to punchline index
                            if 'punchline_vectors' in data and len(data['punchline_vectors']) > 0:
                                punchline_vectors = data['punchline_vectors']
                                current_idx = self.punchline_index.ntotal
                                self.punchline_index.add(punchline_vectors)
                                for i in range(len(punchline_vectors)):
                                    self.punchline_map[current_idx + i] = (bit_id, f"punchline_{i}")

                            if len(vectors_batch) >= batch_size:
                                self._add_vectors_batch(vectors_batch, ids_batch)
                                vectors_batch = []
                                ids_batch = []
                except (np.lib.format.FormatError, OSError) as e:
                    logger.error(f"Error loading vectors for bit {bit_id}: {e}")
                    continue

            # Add remaining vectors
            if vectors_batch:
                self._add_vectors_batch(vectors_batch, ids_batch)

            logger.info("Initialized FAISS indices")

        except RuntimeError as e:
            logger.error(f"Error initializing indices: {e}")
            raise

    def _load_all_vectors(self) -> None:
        """Load all bit vectors and rebuild FAISS indices."""
        try:
            # Clear existing maps
            self.sentence_map.clear()
            self.ngram_map.clear()
            self.punchline_map.clear()

            # Initialize collection arrays
            full_vectors = []
            sentence_vectors = []
            ngram_vectors = []
            punchline_vectors = []

            # Load vectors from central DB
            for file in os.listdir(self.storage.vectors_dir):
                if not file.endswith(".npz"):
                    continue

                bit_id = os.path.splitext(file)[0]
                vector_file = os.path.join(self.storage.vectors_dir, file)

                try:
                    # Load and round vectors
                    data = np.load(vector_file)
                    full_vectors.append(round_vector(data['full_vector']))

                    # Process sentence vectors
                    for sent_idx, sent_vec in enumerate(data['sentence_vectors']):
                        sentence_vectors.append(round_vector(sent_vec))
                        self.sentence_map[sent_idx] = (bit_id, str(sent_vec))

                    # Process n-gram vectors
                    if 'ngram_texts' in data and 'ngram_vectors' in data:
                        for ng_text, ng_vec in zip(data['ngram_texts'], data['ngram_vectors']):
                            ngram_vectors.append(round_vector(ng_vec))
                            self.ngram_map[len(ngram_vectors) - 1] = (bit_id, ng_text)

                    # Process punchline vectors
                    if 'punchline_texts' in data and 'punchline_vectors' in data:
                        for p_text, p_vec in zip(data['punchline_texts'], data['punchline_vectors']):
                            punchline_vectors.append(round_vector(p_vec))
                            self.punchline_map[len(punchline_vectors) - 1] = (bit_id, p_text)

                    logger.info(f"Loaded vectors for bit: {bit_id}")

                except Exception as e:
                    logger.error(f"Error loading vectors for bit {bit_id}: {e}")
                    continue

            # Reset and rebuild indices
            self._init_indices()

            # Add vectors to indices
            if full_vectors:
                self.full_index.add(np.array(full_vectors))
                faiss.write_index(self.full_index, os.path.join(self.storage.indices_dir, "full_index.bin"))

            if sentence_vectors:
                self.sentence_index.add(np.array(sentence_vectors))
                faiss.write_index(self.sentence_index, os.path.join(self.storage.indices_dir, "sentence_index.bin"))

            if ngram_vectors:
                self.ngram_index.add(np.array(ngram_vectors))
                faiss.write_index(self.ngram_index, os.path.join(self.storage.indices_dir, "ngram_index.bin"))

            if punchline_vectors:
                self.punchline_index.add(np.array(punchline_vectors))
                faiss.write_index(self.punchline_index, os.path.join(self.storage.indices_dir, "punchline_index.bin"))

            logger.info(f"Rebuilt FAISS indices with {len(full_vectors)} bits")

        except Exception as e:
            logger.error(f"Error loading vectors and rebuilding indices: {e}")
            raise

    def _add_vectors_batch(self, vectors: List[np.ndarray], bit_ids: List[str]):
        """Add a batch of vectors to indices."""
        if not vectors:
            return

        # Stack vectors and normalize
        vectors_array = np.stack(vectors)
        faiss.normalize_L2(vectors_array)

        # Add to indices
        self.full_index.add(vectors_array)
        # Add to other indices as needed...

    def _generate_bit_id(self, bit_data: Dict[str, Any]) -> str:
        """Generate a unique bit ID."""
        try:
            show_info = bit_data.get('show_info', {})
            show_id = show_info.get('show_identifier')
            bit_info = bit_data.get('bit_info', {})
            bit_title = bit_info.get('title', '').lower().replace(' ', '_')

            if not show_id or not bit_title:
                logger.warning("Missing show identifier or bit title, using UUID")
                return str(uuid.uuid4())

            return f"{show_id}_{bit_title}"

        except (KeyError, AttributeError) as e:
            logger.error(f"Error generating bit ID: {e}")
            return str(uuid.uuid4())

    def close(self):
        """Explicitly close and cleanup resources if needed."""
        self.full_index = None
        self.sentence_index = None
        self.ngram_index = None
        self.punchline_index = None
    
    def add_to_database(self, bit_id: str, bit_data: Dict[str, Any], vectors: BitVectors) -> str:
        """Add a bit to the database."""
        try:
            # Generate bit ID if not provided
            if not bit_id:
                bit_id = self._generate_bit_id(bit_data)

            # Update registry
            self.registry[bit_id] = bit_data
            self.storage.save_registry(self.registry)

            # Save vectors and update indices
            self._add_bit_vectors_to_indices(bit_id, vectors)

            logger.info(f"Added bit {bit_id} to database")
            return bit_id

        except Exception as e:
            logger.error(f"Error adding bit to database: {e}")
            # Clean up any partial saves
            if bit_id in self.registry:
                del self.registry[bit_id]
                self.storage.save_registry(self.registry)
            self.storage.delete_bit_data(bit_id)
            raise

    def update_bit(self, bit_id: str, bit_data: Dict[str, Any], vectors: Optional[BitVectors] = None) -> bool:
        """Update a bit in the database with validation."""
        try:
            # Validate bit exists
            if bit_id not in self.registry:
                raise ValueError(f"Invalid bit_id: {bit_id}")

            # Update registry
            self.registry[bit_id].update({
                'bit_info': bit_data.get('bit_info', {}),
                'show_info': bit_data.get('show_info', {})
            })
            self.storage.save_registry(self.registry)

            # Update vectors if provided
            if vectors is not None:
                # Validate dimensions
                if vectors.full_vector.shape[0] != self.dimension:
                    raise ValueError(
                        f"Full vector dimension mismatch: "
                        f"expected {self.dimension}, got {vectors.full_vector.shape[0]}"
                    )

                # Save new vectors and rebuild indices
                self.storage.save_bit_vectors(bit_id, vectors)
                self._load_all_vectors()

            logger.info(f"Updated bit {bit_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating bit {bit_id}: {e}")
            raise

    def compare(self, bit_data: Dict[str, Any], vectors: BitVectors) -> str:
        """
        Compare a bit with the database and return the standardized title.
        If no match is found, adds the bit to the database.

        Args:
            bit_data: Combined bit data including show info and bit info
            vectors: Vector representations for the bit

        Returns:
            Standardized title for the bit
        """
        matches = self.find_matching_bits(vectors)

        if matches and matches[0].overall_score > BitVectorDBConfig.HARD_MATCH_THRESHOLD:
            # Use existing title
            return matches[0].title

        # Check for soft match
        if matches and matches[0].overall_score > BitVectorDBConfig.SOFT_MATCH_THRESHOLD:
            # Use existing title with soft match indication
            return f"{matches[0].title} (soft match)"

        # No match found, add to database with original title
        self.add_to_database(None, bit_data, vectors)
        return bit_data['bit_info']['title']

    def find_matching_bits(self, query_vectors: BitVectors) -> List[BitMatch]:
        """Find matching bits using multi-level comparison."""
        try:
            logger.info("Starting bit matching...")
            
            # Round query vectors for consistency
            query_full = round_vector(query_vectors.full_vector).reshape(1, -1)
            query_sentences = np.array([round_vector(vec) for vec in query_vectors.sentence_vectors])
            
            # Log index stats
            logger.info(f"Database stats:")
            logger.info(f"- Total bits in registry: {len(self.registry)}")
            logger.info(f"- Full index size: {self.full_index.ntotal}")
            logger.info(f"- Sentence index size: {self.sentence_index.ntotal}")
            logger.info(f"- N-gram index size: {self.ngram_index.ntotal}")
            logger.info(f"- Punchline index size: {self.punchline_index.ntotal}")

            # Get overall similarity with more candidates
            D, I = self.full_index.search(query_full, BitVectorDBConfig.SEARCH['initial_candidates'])  # Search more candidates
            logger.info(f"Found {len(I[0])} initial matches")

            matches = []
            for dist, idx in zip(D[0], I[0]):
                # Skip if distance is way too high
                if dist > BitVectorDBConfig.THRESHOLDS['full_vector']:  # More lenient initial threshold
                    logger.debug(f"Skipping match with distance {dist:.3f} (too high)")
                    continue

                bit_id = list(self.registry.keys())[idx]
                bit_data = self.registry[bit_id]
                logger.info(f"\nChecking match: {bit_id} (distance: {dist:.3f})")

                # Find matching sentences
                matching_sentences = []
                sent_D, sent_I = self.sentence_index.search(query_sentences, BitVectorDBConfig.SEARCH['sentence_candidates'])
                
                for query_idx, (sent_dists, sent_idxs) in enumerate(zip(sent_D, sent_I)):
                    # More stringent threshold for sentences
                    sent_threshold = BitVectorDBConfig.COMPONENT_THRESHOLDS['sentence']['early'] if query_idx < 3 else BitVectorDBConfig.COMPONENT_THRESHOLDS['sentence']['late']
                    
                    for sent_dist, sent_idx in zip(sent_dists, sent_idxs):
                        if sent_dist < sent_threshold:
                            if sent_idx in self.sentence_map:
                                match_bit_id, match_sent = self.sentence_map[sent_idx]
                                if match_bit_id == bit_id:
                                    matching_sentences.append((
                                        str(query_vectors.sentence_vectors[query_idx]),
                                        str(match_sent),
                                        float(sent_dist)
                                    ))

                # Find matching n-grams
                matching_ngrams = []
                for i, (query_ng, query_vec) in enumerate(query_vectors.ngram_vectors):
                    query_vec = round_vector(query_vec).reshape(1, -1)
                    ng_D, ng_I = self.ngram_index.search(query_vec, BitVectorDBConfig.SEARCH['ngram_candidates'])
                    
                    for ng_dist, ng_idx in zip(ng_D[0], ng_I[0]):
                        if ng_dist < BitVectorDBConfig.COMPONENT_THRESHOLDS['ngram']:  # More stringent n-gram threshold
                            if ng_idx in self.ngram_map:
                                match_bit_id, match_ng = self.ngram_map[ng_idx]
                                if match_bit_id == bit_id:
                                    matching_ngrams.append((
                                        str(query_ng),
                                        str(match_ng),
                                        float(ng_dist)
                                    ))

                # Find matching punchlines
                matching_punchlines = []
                for query_p, query_vec, query_weight in query_vectors.punchline_vectors:
                    query_vec = round_vector(query_vec).reshape(1, -1)
                    p_D, p_I = self.punchline_index.search(query_vec, BitVectorDBConfig.SEARCH['punchline_candidates'])
                    
                    for p_dist, p_idx in zip(p_D[0], p_I[0]):
                        if p_dist < BitVectorDBConfig.COMPONENT_THRESHOLDS['punchline']:  # Adjusted punchline threshold
                            if p_idx in self.punchline_map:
                                match_bit_id, match_p = self.punchline_map[p_idx]
                                if match_bit_id == bit_id:
                                    matching_punchlines.append((
                                        str(query_p),
                                        str(match_p),
                                        float(p_dist)
                                    ))

                # Component weights - use config values
                weights = dict(BitVectorDBConfig.WEIGHTS)
                
                # Boost weights for strong component matches
                if dist < BitVectorDBConfig.BOOST_THRESHOLDS['full_vector']:
                    weights['full'] *= BitVectorDBConfig.BOOST_FACTORS['full']
                if matching_sentences and all(s[2] < BitVectorDBConfig.BOOST_THRESHOLDS['sentences'] for s in matching_sentences):
                    weights['sent'] *= BitVectorDBConfig.BOOST_FACTORS['sent']
                if matching_ngrams and all(n[2] < BitVectorDBConfig.BOOST_THRESHOLDS['ngrams'] for n in matching_ngrams):
                    weights['ngram'] *= BitVectorDBConfig.BOOST_FACTORS['ngram']
                if matching_punchlines and all(p[2] < BitVectorDBConfig.BOOST_THRESHOLDS['punchlines'] for p in matching_punchlines):
                    weights['punch'] *= BitVectorDBConfig.BOOST_FACTORS['punch']
                    
                # Normalize weights
                total = sum(weights.values())
                weights = {k: v/total for k, v in weights.items()}
                
                # Calculate component scores
                full_score = 1 - dist
                
                # Sentence score focuses on best matches with position weighting
                if matching_sentences:
                    sent_scores = [s[2] for s in matching_sentences]
                    sent_scores.sort()
                    best_sent_scores = sent_scores[:min(3, len(sent_scores))]
                    # Weight earlier sentences more heavily
                    position_weights = np.array([1.2, 1.0, 0.8])[:len(best_sent_scores)]
                    position_weights = position_weights / position_weights.sum()
                    sentence_score = 1 - np.average(best_sent_scores, weights=position_weights)
                else:
                    sentence_score = 0
                
                # N-gram score weighted by position
                if matching_ngrams:
                    ng_scores = [n[2] for n in matching_ngrams]
                    ng_scores.sort()
                    best_ng_scores = ng_scores[:min(2, len(ng_scores))]
                    ngram_score = 1 - np.mean(best_ng_scores)
                else:
                    ngram_score = 0
                
                # Punchline score emphasizes best match
                if matching_punchlines:
                    punch_scores = [p[2] for p in matching_punchlines]
                    punch_scores.sort()
                    punchline_score = 1 - punch_scores[0]  # Best match only
                else:
                    punchline_score = 0

                # Calculate weighted score
                overall_score = (
                    full_score * weights['full'] +
                    sentence_score * weights['sent'] +
                    ngram_score * weights['ngram'] +
                    punchline_score * weights['punch']
                )
                
                # Boost score if we have multiple strong components
                strong_components = sum(1 for score in [
                    full_score, sentence_score, ngram_score, punchline_score
                ] if score > BitVectorDBConfig.BOOST_THRESHOLDS['multi_component'])
                
                if strong_components >= 2:
                    overall_score *= BitVectorDBConfig.MULTI_COMPONENT_BOOST

                logger.info(f"Match scores for {bit_id}:")
                logger.info(f"- Full vector ({weights['full']:.2f}): {full_score:.3f}")
                logger.info(f"- Sentences ({weights['sent']:.2f}): {sentence_score:.3f}")
                logger.info(f"- N-grams ({weights['ngram']:.2f}): {ngram_score:.3f}")
                logger.info(f"- Punchlines ({weights['punch']:.2f}): {punchline_score:.3f}")
                logger.info(f"- Overall: {overall_score:.3f}")
                
                match = BitMatch(
                    bit_id=bit_id,
                    title=bit_data.get('bit_info', {}).get('title', bit_id),
                    overall_score=overall_score,
                    sentence_matches=matching_sentences,
                    ngram_matches=matching_ngrams,
                    punchline_matches=matching_punchlines,
                    show_info=bit_data.get('show_info'),
                    match_type="exact" if overall_score > BitVectorDBConfig.HARD_MATCH_THRESHOLD else "soft"
                )
                matches.append(match)

            # Sort matches by overall score
            matches.sort(key=lambda x: x.overall_score, reverse=True)
            logger.info(f"\nFound {len(matches)} total matches")
            for m in matches[:3]:
                logger.info(f"Top match: {m.title} (score: {m.overall_score:.3f}, type: {m.match_type})")

            return matches

        except Exception as e:
            logger.error(f"Error finding matching bits: {e}")
            raise

    def run(self) -> None:
        """Required method to satisfy BaseTool abstract class."""
        # Load vectors and build indices
        self._load_all_vectors()
        logger.info(f"Loaded vectors for {len(self.registry)} bits")
