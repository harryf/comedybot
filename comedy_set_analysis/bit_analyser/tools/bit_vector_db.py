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
from bit_entity import BitEntity

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
    
    # L2 distance thresholds (0 = identical, 2 = orthogonal)
    THRESHOLDS = {
        'full_vector': 1.0,   # More lenient for full vector
        'sentences': 1.2,     # More lenient for sentences
        'ngrams': 1.0,        # Strict for ngrams
        'punchlines': 1.0     # Strict for punchlines
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
        self.bits_dir = os.path.join(self.base_dir, 'bits/')
        self.registry_file = os.path.join(self.base_dir, 'bit_registry.json')
        self.canonical_bits_file = os.path.join(self.base_dir, 'canonical_bits.json')
        self._init_directories()

    def _init_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.base_dir, self.vectors_dir, self.indices_dir, self.bits_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied creating directory {directory}: {e}")
                raise
            except OSError as e:
                logger.error(f"OS error creating directory {directory}: {e}")
                raise

    def load_registry(self) -> Dict[str, str]:
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

    def save_registry(self, registry: Dict[str, str]):
        """Save bit registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def save_bit_vectors(self, bit_id: str, vectors: BitVectors) -> str:
        """Save a bit's vectors to disk."""
        try:
            # Ensure vectors directory exists
            os.makedirs(self.vectors_dir, exist_ok=True)
            
            # Save vectors to npz file
            vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            
            # Extract n-gram components
            ngram_texts = []
            ngram_vectors = []
            ngram_positions = []
            if vectors.ngram_vectors:  # Only process if not empty
                for text, vec, pos in vectors.ngram_vectors:
                    ngram_texts.append(str(text))  # Ensure text is string
                    ngram_vectors.append(vec)
                    ngram_positions.append(int(pos))  # Ensure position is int

            # Extract punchline components
            punchline_texts = []
            punchline_vectors = []
            punchline_weights = []
            if vectors.punchline_vectors:  # Only process if not empty
                for text, vec, weight in vectors.punchline_vectors:
                    punchline_texts.append(str(text))  # Ensure text is string
                    punchline_vectors.append(vec)
                    punchline_weights.append(float(weight))  # Ensure weight is float

            # Save to npz file with proper type handling
            np.savez(
                vector_file,
                full_vector=round_vector(vectors.full_vector),
                sentence_vectors=np.array([round_vector(vec) for vec in vectors.sentence_vectors]) if vectors.sentence_vectors else np.array([], dtype=np.float32),
                ngram_texts=np.array(ngram_texts, dtype=object) if ngram_texts else np.array([], dtype=object),
                ngram_vectors=np.array([round_vector(vec) for vec in ngram_vectors]) if ngram_vectors else np.array([], dtype=np.float32),
                ngram_positions=np.array(ngram_positions, dtype=np.int32) if ngram_positions else np.array([], dtype=np.int32),
                punchline_texts=np.array(punchline_texts, dtype=object) if punchline_texts else np.array([], dtype=object),
                punchline_vectors=np.array([round_vector(vec) for vec in punchline_vectors]) if punchline_vectors else np.array([], dtype=np.float32),
                punchline_weights=np.array(punchline_weights, dtype=np.float32) if punchline_weights else np.array([], dtype=np.float32)
            )
            
            return vector_file
            
        except Exception as e:
            logger.error(f"Error saving vectors for bit {bit_id}: {e}")
            raise

    def load_bit_vectors(self, bit_id: str) -> Optional[BitVectors]:
        """Load a bit's vectors from disk.
        
        Returns:
            BitVectors object if successful, None if loading fails
        """
        try:
            vector_file = os.path.join(self.vectors_dir, f"{bit_id}.npz")
            if not os.path.exists(vector_file):
                logger.error(f"Vector file not found for bit {bit_id}")
                return None
                
            with np.load(vector_file, allow_pickle=True) as data:
                # Handle empty arrays for ngrams
                ngram_tuples = []
                if data['ngram_texts'].size > 0:
                    try:
                        for text, vec, pos in zip(data['ngram_texts'], data['ngram_vectors'], data['ngram_positions']):
                            ngram_tuples.append((str(text), vec, int(pos)))
                    except Exception as e:
                        logger.warning(f"Error reconstructing ngram tuples for bit {bit_id}: {e}")

                # Handle empty arrays for punchlines
                punchline_tuples = []
                if data['punchline_texts'].size > 0:
                    try:
                        for text, vec, weight in zip(data['punchline_texts'], data['punchline_vectors'], data['punchline_weights']):
                            punchline_tuples.append((str(text), vec, float(weight)))
                    except Exception as e:
                        logger.warning(f"Error reconstructing punchline tuples for bit {bit_id}: {e}")

                # Return a proper BitVectors object
                return BitVectors(
                    full_vector=data['full_vector'],
                    sentence_vectors=data['sentence_vectors'],
                    ngram_vectors=ngram_tuples,
                    punchline_vectors=punchline_tuples
                )
                
        except Exception as e:
            logger.error(f"Error loading vectors for bit {bit_id}: {e}")
            return None

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

    def load_canonical_bits(self) -> Dict[str, List[str]]:
        """Load canonical bits mapping from file.
        
        Returns:
            Dict mapping canonical bit names to lists of bit IDs.
            Example:
            {
                "Not Breeding": [
                    "20241102_The_Comedy_Clubhouse_not_breeding",
                    "20241231_Skyline_Conedy_not_breeding"
                ]
            }
        """
        try:
            if os.path.exists(self.canonical_bits_file):
                with open(self.canonical_bits_file, 'r') as f:
                    return json.load(f)
            return {}
        except FileNotFoundError:
            logger.error(f"Canonical bits file not found: {self.canonical_bits_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing canonical bits file: {e}")
            return {}
        except PermissionError:
            logger.error(f"Permission denied accessing canonical bits file: {self.canonical_bits_file}")
            return {}

    def save_canonical_bits(self, canonical_bits: Dict[str, List[str]]):
        """Save canonical bits mapping to file.
        
        Args:
            canonical_bits: Dict mapping canonical bit names to lists of bit IDs.
            Example:
            {
                "Not Breeding": [
                    "20241102_The_Comedy_Clubhouse_not_breeding",
                    "20241231_Skyline_Conedy_not_breeding"
                ]
            }
        """
        try:
            with open(self.canonical_bits_file, 'w') as f:
                json.dump(canonical_bits, f, indent=2)
            logger.info("Saved canonical bits mapping")
        except (OSError, IOError) as e:
            logger.error(f"Error saving canonical bits: {e}")
            raise
    
    def save_bit(self, bit_entity: BitEntity):
        """Save a bit to the storage."""
        try:
            bit_entity.write_to_database(self.bits_dir)
            logger.info(f"Saved bit {bit_entity.bit_data['bit_id']} to {self.bits_dir}")
        except Exception as e:
            logger.error(f"Error saving bit: {e}")
            raise
    
    def load_bit(self, bit_id: str) -> Optional[BitEntity]:
        """Load a bit from the storage."""
        try:
            bit_entity = BitEntity(self.bits_dir)
            bit_entity.load_from_database(bit_id)
            return bit_entity
        except Exception as e:
            logger.error(f"Error loading bit: {e}")
            return None

    def load_data(self, filename: str) -> Dict[str, Any]:
        """Load data from a file."""
        try:
            file_path = os.path.join(self.base_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing file: {e}")
            return {}
        except PermissionError:
            logger.error(f"Permission denied accessing file: {file_path}")
            return {}

    def save_data(self, filename: str, data: Dict[str, Any]):
        """Save data to a file."""
        try:
            file_path = os.path.join(self.base_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data to {file_path}")
        except (OSError, IOError) as e:
            logger.error(f"Error saving data: {e}")
            raise

class CanonicalBits:
    """Manages the mapping of canonical bit names to their various versions."""

    def __init__(self, storage_manager: BitStorageManager):
        """Initialize with a storage manager instance."""
        self.storage = storage_manager
        self.canonical_map = self.storage.load_canonical_bits()

    def save(self):
        """Save the current canonical bits mapping to storage."""
        self.storage.save_canonical_bits(self.canonical_map)

    def add_bit(self, bit_title: str, bit_id: str, matching_bit_id: Optional[str] = None):
        """Add a bit to the canonical mapping.
        
        Args:
            bit_title: The title of the bit to add
            bit_id: The ID of the bit to add
            matching_bit_id: Optional ID of a matching bit to group with
        
        The method follows these steps:
        1. If bit_id already exists anywhere in the mapping, return
        2. If bit_title exists as a key, add bit_id to its list
        3. If matching_bit_id is provided and exists, add bit_id to that bit's group
        4. If none of the above, create a new entry with bit_title as key
        """
        # Check if bit_id already exists anywhere
        for bit_ids in self.canonical_map.values():
            if bit_id in bit_ids:
                logger.info(f"Bit ID {bit_id} already exists in canonical mapping")
                return

        # Check if bit_title exists as a key
        if bit_title in self.canonical_map:
            self.canonical_map[bit_title].append(bit_id)
            logger.info(f"Added {bit_id} to existing canonical title '{bit_title}'")
            return

        # If matching_bit_id provided, look for it
        if matching_bit_id is not None:
            for title, bit_ids in self.canonical_map.items():
                if matching_bit_id in bit_ids:
                    bit_ids.append(bit_id)
                    logger.info(f"Added {bit_id} to group with matching bit {matching_bit_id}")
                    return

        # Create new entry
        new_bits = [bit_id]
        if matching_bit_id is not None:
            new_bits.append(matching_bit_id)
        self.canonical_map[bit_title] = new_bits
        logger.info(f"Created new canonical entry '{bit_title}' with bits {new_bits}")

    def get_bit_by_title(self, title: str) -> Optional[Tuple[str, List[str]]]:
        """Get a bit by its exact title.
        
        Args:
            title: The exact title to search for
            
        Returns:
            A tuple of (title, list of bit_ids) if found, None otherwise
        """
        if title in self.canonical_map:
            return (title, self.canonical_map[title])
        return None

    def get_bit_by_id(self, bit_id: str) -> Optional[Tuple[str, List[str]]]:
        """Find a bit by searching for its ID in the values of the canonical map.
        
        Args:
            bit_id: The bit ID to search for
            
        Returns:
            A tuple of (title, list of bit_ids) if found, None otherwise
        """
        for title, bit_ids in self.canonical_map.items():
            if bit_id in bit_ids:
                return (title, bit_ids)
        return None

class JokeTypeTracker:
    """Manages the mapping of joke types to bit IDs."""
    
    def __init__(self, storage_manager: BitStorageManager):
        """Initialize with a storage manager instance."""
        self.storage = storage_manager
        self.joke_type_map = self.storage.load_data('joke_types.json') or {}
    
    def save(self):
        """Save the current joke type mapping to storage."""
        self.storage.save_data('joke_types.json', self.joke_type_map)
    
    def add_bit(self, bit_id: str, joke_types: List[str]):
        """Add a bit's joke types to the mapping."""
        for joke_type in joke_types:
            if joke_type not in self.joke_type_map:
                self.joke_type_map[joke_type] = []
            if bit_id not in self.joke_type_map[joke_type]:
                self.joke_type_map[joke_type].append(bit_id)
        self.save()


class ThemeTracker:
    """Manages the mapping of themes to bit IDs."""
    
    def __init__(self, storage_manager: BitStorageManager):
        """Initialize with a storage manager instance."""
        self.storage = storage_manager
        self.theme_map = self.storage.load_data('themes.json') or {}
    
    def save(self):
        """Save the current theme mapping to storage."""
        self.storage.save_data('themes.json', self.theme_map)
    
    def add_bit(self, bit_id: str, themes: List[str]):
        """Add a bit's themes to the mapping."""
        for theme in themes:
            if theme not in self.theme_map:
                self.theme_map[theme] = []
            if bit_id not in self.theme_map[theme]:
                self.theme_map[theme].append(bit_id)
        self.save()

class BitMatch(BaseModel):
    """
    Represents a matching bit from the database.
    """
    bit_id: str
    title: str
    overall_score: float
    sentence_matches: List[Tuple[str, str, float]] = Field(default_factory=list)
    ngram_matches: List[Tuple[str, str, float, int]] = Field(default_factory=list)
    punchline_matches: List[Tuple[str, str, float]] = Field(default_factory=list)
    show_info: Optional[Dict[str, Any]] = None
    match_type: str = Field(default="exact")  # "exact", "soft", or "none"

class BitVectorDB:
    """Database for Comedy Bits with FAISS indexing."""

    def __init__(self, dimension: int = 384, similarity_threshold: float = 0.5):
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

            # Load or initialize registry as a dict of bit_id -> timestamp
            self.registry = self.storage.load_registry() or {}

            self.canonical_bits = CanonicalBits(self.storage)
            self.joke_type_tracker = JokeTypeTracker(self.storage)
            self.theme_tracker = ThemeTracker(self.storage)

            # Initialize FAISS indices
            self._init_indices()

            logger.info(f"Initialized BitDB with dimension: {dimension}, threshold: {similarity_threshold}")

        except RuntimeError as e:
            logger.error(f"FAISS initialization error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid parameter value: {e}")
            raise

    def _load_registry(self) -> Dict[str, str]:
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

    def _add_bit_vectors_to_indices(self, bit_id: str, vectors: Optional[BitVectors]):
        """Add a bit's vectors to the indices."""
        if vectors is None:
            logger.warning(f"No vectors provided for bit {bit_id}")
            return

        try:
            # Add full vector
            logger.debug(f"Adding full vector for bit {bit_id}")
            if vectors.full_vector is not None:
                self.full_index.add(vectors.full_vector.reshape(1, -1))
            else:
                logger.warning(f"No full vector for bit {bit_id}")

            # Add sentence vectors
            logger.debug(f"Adding {len(vectors.sentence_vectors)} sentence vectors for bit {bit_id}")
            if vectors.sentence_vectors:
                sentence_vectors = np.array([vec for vec in vectors.sentence_vectors])
                self.sentence_index.add(sentence_vectors)
                # Map indices to bit_id only
                start_idx = self.sentence_index.ntotal - len(sentence_vectors)
                for i in range(len(sentence_vectors)):
                    self.sentence_map[start_idx + i] = bit_id
            else:
                logger.debug(f"No sentence vectors for bit {bit_id}")

            # Add ngram vectors
            logger.debug(f"Processing {len(vectors.ngram_vectors)} ngram vectors for bit {bit_id}")
            if vectors.ngram_vectors:
                try:
                    ngram_vectors = []
                    ngram_texts = []
                    ngram_positions = []
                    for text, vec, pos in vectors.ngram_vectors:
                        logger.debug(f"Processing ngram: text='{text}', pos={pos}, vec_shape={vec.shape if vec is not None else 'None'}")
                        if vec is not None:
                            ngram_vectors.append(vec)
                            ngram_texts.append(text)
                            ngram_positions.append(pos)
                        else:
                            logger.warning(f"None vector found for ngram '{text}' at position {pos}")
                    
                    if ngram_vectors:
                        ngram_vectors = np.array(ngram_vectors)
                        logger.debug(f"Adding {len(ngram_vectors)} ngram vectors to index")
                        self.ngram_index.add(ngram_vectors)
                        # Map indices to bit_id
                        start_idx = self.ngram_index.ntotal - len(ngram_vectors)
                        for i in range(len(ngram_vectors)):
                            self.ngram_map[start_idx + i] = (bit_id, ngram_texts[i], ngram_positions[i])
                    else:
                        logger.warning(f"No valid ngram vectors to add for bit {bit_id}")
                except Exception as e:
                    logger.error(f"Error processing ngram vectors for bit {bit_id}: {e}")
                    raise
            else:
                logger.debug(f"No ngram vectors for bit {bit_id}")

            # Add punchline vectors
            logger.debug(f"Processing {len(vectors.punchline_vectors)} punchline vectors for bit {bit_id}")
            if vectors.punchline_vectors:
                try:
                    punchline_vectors = []
                    punchline_texts = []
                    punchline_weights = []
                    for text, vec, weight in vectors.punchline_vectors:
                        logger.debug(f"Processing punchline: text='{text}', weight={weight}, vec_shape={vec.shape if vec is not None else 'None'}")
                        if vec is not None:
                            punchline_vectors.append(vec)
                            punchline_texts.append(text)
                            punchline_weights.append(weight)
                        else:
                            logger.warning(f"None vector found for punchline '{text}' with weight {weight}")
                    
                    if punchline_vectors:
                        punchline_vectors = np.array(punchline_vectors)
                        logger.debug(f"Adding {len(punchline_vectors)} punchline vectors to index")
                        self.punchline_index.add(punchline_vectors)
                        # Map indices to bit_id
                        start_idx = self.punchline_index.ntotal - len(punchline_vectors)
                        for i in range(len(punchline_vectors)):
                            self.punchline_map[start_idx + i] = (bit_id, punchline_texts[i])
                    else:
                        logger.warning(f"No valid punchline vectors to add for bit {bit_id}")
                except Exception as e:
                    logger.error(f"Error processing punchline vectors for bit {bit_id}: {e}")
                    raise
            else:
                logger.debug(f"No punchline vectors for bit {bit_id}")

        except Exception as e:
            logger.error(f"Error adding vectors for bit {bit_id}: {e}")
            raise

    def find_matching_bits(self, query_vectors: BitVectors) -> List[BitMatch]:
        """Find matching bits using multi-level comparison."""
        try:
            matches = {}
            
            # Search full vector index
            k = min(BitVectorDBConfig.SEARCH['initial_candidates'], self.full_index.ntotal)
            if k == 0:
                return []
                
            D, I = self.full_index.search(query_vectors.full_vector.reshape(1, -1), k)
            
            # Process each candidate
            for idx, distance in zip(I[0], D[0]):
                if distance > BitVectorDBConfig.THRESHOLDS['full_vector']:
                    continue
                    
                bit_id = list(self.registry.keys())[idx]  # Get bit_id by index
                
                # Initialize match object
                match = BitMatch(
                    bit_id=bit_id,
                    title=bit_id,  # Will be updated later if canonical name exists
                    overall_score=0.0
                )

                # Calculate scores (convert L2 distance to similarity score)
                full_score = max(0, 1.0 - (distance / 2.0))  # Normalize to [0,1]
                
                # Search sentence vectors
                sent_score = 0.0
                if query_vectors.sentence_vectors and self.sentence_index.ntotal > 0:
                    k_sent = min(BitVectorDBConfig.SEARCH['sentence_candidates'], self.sentence_index.ntotal)
                    for query_sent in query_vectors.sentence_vectors:
                        D_sent, I_sent = self.sentence_index.search(query_sent.reshape(1, -1), k_sent)
                        for sent_idx, sent_dist in zip(I_sent[0], D_sent[0]):
                            if sent_dist > BitVectorDBConfig.THRESHOLDS['sentences']:
                                continue
                            match_bit_id = self.sentence_map[sent_idx]
                            if match_bit_id == bit_id:
                                score = max(0, 1.0 - (sent_dist / 2.0))
                                sent_score = max(sent_score, score)
                                match.sentence_matches.append(("", "", score))

                # Search ngram vectors
                ngram_score = 0.0
                if query_vectors.ngram_vectors and self.ngram_index.ntotal > 0:
                    k_ng = min(BitVectorDBConfig.SEARCH['ngram_candidates'], self.ngram_index.ntotal)
                    for text, ng_vec, pos in query_vectors.ngram_vectors:
                        if ng_vec is not None:
                            D_ng, I_ng = self.ngram_index.search(ng_vec.reshape(1, -1), k_ng)
                            for ng_idx, ng_dist in zip(I_ng[0], D_ng[0]):
                                if ng_dist > BitVectorDBConfig.THRESHOLDS['ngrams']:
                                    continue
                                match_bit_id, match_text, match_pos = self.ngram_map[ng_idx]
                                if match_bit_id == bit_id:
                                    score = max(0, 1.0 - (ng_dist / 2.0))
                                    ngram_score = max(ngram_score, score)
                                    match.ngram_matches.append((match_text, text, score, match_pos))

                # Search punchline vectors
                punch_score = 0.0
                if query_vectors.punchline_vectors and self.punchline_index.ntotal > 0:
                    k_punch = min(BitVectorDBConfig.SEARCH['punchline_candidates'], self.punchline_index.ntotal)
                    for text, punch_vec, _ in query_vectors.punchline_vectors:
                        if punch_vec is not None:
                            D_punch, I_punch = self.punchline_index.search(punch_vec.reshape(1, -1), k_punch)
                            for punch_idx, punch_dist in zip(I_punch[0], D_punch[0]):
                                if punch_dist > BitVectorDBConfig.THRESHOLDS['punchlines']:
                                    continue
                                match_bit_id, match_text = self.punchline_map[punch_idx]
                                if match_bit_id == bit_id:
                                    score = max(0, 1.0 - (punch_dist / 2.0))
                                    punch_score = max(punch_score, score)
                                    match.punchline_matches.append((match_text, text, score))

                # Calculate overall score with component weights
                match.overall_score = (
                    BitVectorDBConfig.WEIGHTS['full'] * full_score +
                    BitVectorDBConfig.WEIGHTS['sent'] * sent_score +
                    BitVectorDBConfig.WEIGHTS['ngram'] * ngram_score +
                    BitVectorDBConfig.WEIGHTS['punch'] * punch_score
                )

                # Apply boost factors for strong component matches
                boost = 1.0
                if full_score > BitVectorDBConfig.BOOST_THRESHOLDS['full_vector']:
                    boost *= BitVectorDBConfig.BOOST_FACTORS['full']
                if sent_score > BitVectorDBConfig.BOOST_THRESHOLDS['sentences']:
                    boost *= BitVectorDBConfig.BOOST_FACTORS['sent']
                if ngram_score > BitVectorDBConfig.BOOST_THRESHOLDS['ngrams']:
                    boost *= BitVectorDBConfig.BOOST_FACTORS['ngram']
                if punch_score > BitVectorDBConfig.BOOST_THRESHOLDS['punchlines']:
                    boost *= BitVectorDBConfig.BOOST_FACTORS['punch']

                match.overall_score *= boost

                # Log component scores
                logger.info(f"Match scores for {bit_id}:")
                logger.info(f"- Full vector ({BitVectorDBConfig.WEIGHTS['full']:.2f}): {full_score:.3f}")
                logger.info(f"- Sentences ({BitVectorDBConfig.WEIGHTS['sent']:.2f}): {sent_score:.3f}")
                logger.info(f"- N-grams ({BitVectorDBConfig.WEIGHTS['ngram']:.2f}): {ngram_score:.3f}")
                logger.info(f"- Punchlines ({BitVectorDBConfig.WEIGHTS['punch']:.2f}): {punch_score:.3f}")
                logger.info(f"- Overall: {match.overall_score:.3f}")

                matches[bit_id] = match

            # Sort matches by overall score
            sorted_matches = sorted(
                matches.values(),
                key=lambda x: x.overall_score,
                reverse=True
            )

            # Filter matches below threshold
            filtered_matches = [
                match for match in sorted_matches
                if match.overall_score >= self.similarity_threshold
            ]

            logger.info(f"\nFound {len(filtered_matches)} total matches")
            if filtered_matches:
                top_match = filtered_matches[0]
                logger.info(f"Top match: {top_match.title} (score: {top_match.overall_score:.3f}, type: {top_match.match_type})")

            return filtered_matches

        except Exception as e:
            logger.error(f"Error finding matching bits: {e}")
            raise

    def run(self) -> None:
        """Required method to satisfy BaseTool abstract class."""
        # Load vectors and build indices
        self._load_all_vectors()
        logger.info(f"Loaded vectors for {len(self.registry)} bits")

    def _load_all_vectors(self) -> None:
        """Load all bit vectors and rebuild FAISS indices."""
        try:
            # Clear existing maps
            self.sentence_map.clear()
            self.ngram_map.clear()
            self.punchline_map.clear()

            # Initialize collection arrays
            full_vectors = []
            bit_ids = []

            # Load vectors from central DB
            for file in os.listdir(self.storage.vectors_dir):
                if not file.endswith(".npz"):
                    continue

                bit_id = os.path.splitext(file)[0]
                try:
                    with np.load(os.path.join(self.storage.vectors_dir, file)) as data:
                        # Convert to float32, normalize
                        full_vec = round_vector(data['full_vector']).astype(np.float32).reshape(1, -1)
                        faiss.normalize_L2(full_vec)
                        full_vectors.append(full_vec[0])
                        bit_ids.append(bit_id)
                        self.registry[bit_id] = datetime.now().isoformat()

                        # Add to sentence index
                        if 'sentence_vectors' in data and len(data['sentence_vectors']) > 0:
                            sentence_vectors = data['sentence_vectors'].astype(np.float32)
                            faiss.normalize_L2(sentence_vectors)
                            current_idx = self.sentence_index.ntotal
                            self.sentence_index.add(sentence_vectors)
                            for i in range(len(sentence_vectors)):
                                self.sentence_map[current_idx + i] = bit_id

                        # Add to ngram index
                        if 'ngram_vectors' in data and len(data['ngram_vectors']) > 0:
                            try:
                                ngram_tuples = data['ngram_vectors']
                                if len(ngram_tuples) > 0:
                                    ngram_texts = [t[0] for t in ngram_tuples]
                                    ngram_vectors = np.array([t[1] for t in ngram_tuples], dtype=np.float32)
                                    ngram_positions = [t[2] for t in ngram_tuples]
                                    
                                    if ngram_vectors.size > 0:
                                        faiss.normalize_L2(ngram_vectors)
                                        current_idx = self.ngram_index.ntotal
                                        self.ngram_index.add(ngram_vectors)
                                        for i in range(len(ngram_vectors)):
                                            self.ngram_map[current_idx + i] = (bit_id, ngram_texts[i], ngram_positions[i])
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing ngram vectors for bit {bit_id}: {e}")

                        # Add to punchline index
                        if 'punchline_vectors' in data and len(data['punchline_vectors']) > 0:
                            try:
                                punchline_tuples = data['punchline_vectors']
                                if len(punchline_tuples) > 0:
                                    punchline_texts = [t[0] for t in punchline_tuples]
                                    punchline_vectors = np.array([t[1] for t in punchline_tuples], dtype=np.float32)
                                    
                                    if punchline_vectors.size > 0:
                                        faiss.normalize_L2(punchline_vectors)
                                        current_idx = self.punchline_index.ntotal
                                        self.punchline_index.add(punchline_vectors)
                                        for i in range(len(punchline_vectors)):
                                            self.punchline_map[current_idx + i] = (bit_id, punchline_texts[i])
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing punchline vectors for bit {bit_id}: {e}")

                        logger.info(f"Loaded vectors for bit: {bit_id}")

                except Exception as e:
                    logger.error(f"Error loading vectors for bit {bit_id}: {e}")
                    continue

            # Add full vectors in batch
            if full_vectors:
                full_vectors = np.array(full_vectors)
                self.full_index.add(full_vectors)

            logger.info(f"Loaded {len(full_vectors)} bits into full index")
            logger.info(f"Sentence index size: {self.sentence_index.ntotal}")
            logger.info(f"Ngram index size: {self.ngram_index.ntotal}")
            logger.info(f"Punchline index size: {self.punchline_index.ntotal}")

        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
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

            for vector_file in self.storage.list_vector_files():
                bit_id = os.path.splitext(vector_file)[0]
                try:
                    with np.load(os.path.join(self.storage.vectors_dir, vector_file)) as data:

                        # Convert to float32, normalize
                        full_vec = round_vector(data['full_vector']).astype(np.float32).reshape(1, -1)
                        faiss.normalize_L2(full_vec)

                        # Add to full vector batch
                        vectors_batch.append(full_vec[0])
                        ids_batch.append(bit_id)

                        # Add to sentence index
                        if 'sentence_vectors' in data and len(data['sentence_vectors']) > 0:
                            sentence_vectors = data['sentence_vectors'].astype(np.float32)
                            faiss.normalize_L2(sentence_vectors)
                            current_idx = self.sentence_index.ntotal
                            self.sentence_index.add(sentence_vectors)
                            for i in range(len(sentence_vectors)):
                                self.sentence_map[current_idx + i] = bit_id

                        # Add to ngram index
                        if 'ngram_vectors' in data and len(data['ngram_vectors']) > 0:
                            try:
                                ngram_tuples = data['ngram_vectors']
                                if len(ngram_tuples) > 0:
                                    ngram_texts = [t[0] for t in ngram_tuples]
                                    ngram_vectors = np.array([t[1] for t in ngram_tuples], dtype=np.float32)
                                    ngram_positions = [t[2] for t in ngram_tuples]
                                    
                                    if ngram_vectors.size > 0:
                                        faiss.normalize_L2(ngram_vectors)
                                        current_idx = self.ngram_index.ntotal
                                        self.ngram_index.add(ngram_vectors)
                                        for i in range(len(ngram_vectors)):
                                            self.ngram_map[current_idx + i] = (bit_id, ngram_texts[i], ngram_positions[i])
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing ngram vectors for bit {bit_id}: {e}")

                        # Add to punchline index
                        if 'punchline_vectors' in data and len(data['punchline_vectors']) > 0:
                            try:
                                punchline_tuples = data['punchline_vectors']
                                if len(punchline_tuples) > 0:
                                    punchline_texts = [t[0] for t in punchline_tuples]
                                    punchline_vectors = np.array([t[1] for t in punchline_tuples], dtype=np.float32)
                                    
                                    if punchline_vectors.size > 0:
                                        faiss.normalize_L2(punchline_vectors)
                                        current_idx = self.punchline_index.ntotal
                                        self.punchline_index.add(punchline_vectors)
                                        for i in range(len(punchline_vectors)):
                                            self.punchline_map[current_idx + i] = (bit_id, punchline_texts[i])
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error processing punchline vectors for bit {bit_id}: {e}")

                        if len(vectors_batch) >= batch_size:
                            self._add_vectors_batch(vectors_batch, ids_batch)
                            vectors_batch = []
                            ids_batch = []
                except (ValueError, OSError) as e:
                    logger.error(f"Error loading vectors for bit {bit_id}: {e}")

            # Add remaining vectors
            if vectors_batch:
                self._add_vectors_batch(vectors_batch, ids_batch)

            logger.info("Initialized FAISS indices")

        except RuntimeError as e:
            logger.error(f"Error initializing indices: {e}")
            raise

    def _add_vectors_batch(self, vectors: List[np.ndarray], bit_ids: List[str]):
        """Add a batch of vectors to indices."""
        if not vectors:
            return

        # Stack vectors and normalize
        vectors_array = np.stack(vectors).astype(np.float32)
        faiss.normalize_L2(vectors_array)
        self.full_index.add(vectors_array)

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

    def has_bit(self, bit_id: str) -> bool:
        """Check if a bit ID exists in the registry.
        
        Args:
            bit_id: The ID of the bit to check
            
        Returns:
            True if the bit exists in the registry, False otherwise
        """
        return bit_id in self.registry
    
    def add_to_database(self, bit_entity, vectors: BitVectors, match: Optional[BitMatch] = None) -> str:
        """Add a bit to the database."""
        try:
            # Generate bit ID if not provided
            bit_id = bit_entity.bit_data['bit_id']
            
            if bit_id in self.registry:
                logger.warning(f"Bit {bit_id} already exists in database")
                return
            

            # Update registry
            self.registry[bit_id] = datetime.now().isoformat()
            self.storage.save_registry(self.registry)

            bit_title = bit_entity.bit_data['bit_info']['title']
            if match:
                self.canonical_bits.add_bit(
                    match.title,
                    bit_id,
                    match.bit_id
                )
            else:
                self.canonical_bits.add_bit(
                    bit_title,
                    bit_id
                )
            self.canonical_bits.save()

            joke_types = bit_entity.bit_data['bit_info'].get('joke_types', [])
            themes = bit_entity.bit_data['bit_info'].get('themes', [])
            self.joke_type_tracker.add_bit(bit_id, joke_types)
            self.theme_tracker.add_bit(bit_id, themes)

            # Save vectors and update indices
            self._add_bit_vectors_to_indices(bit_id, vectors)

            self.storage.save_bit(bit_entity)

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
