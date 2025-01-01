#!/usr/bin/env python3

"""
Interface for managing a "database" of comedy bit term vectors.
Uses a file-based storage system in ~/.comedybot with the following structure:

~/.comedybot/
├── bit_vectors/          # Directory containing individual bit vector JSON files
│   ├── hash1.json
│   ├── hash2.json
│   └── ...
├── faiss_index.bin      # Binary FAISS index for vector similarity search
└── bit_registry.json    # Registry mapping bit titles to their vector hashes
"""

import os
import sys
import json
import logging
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class BitVectorDB:
    def __init__(self, similarity_threshold: float = 0.85, base_dir: Optional[str] = None):
        """
        Initialize the bit vector database interface.
        
        Args:
            similarity_threshold: Threshold for considering bits similar (default: 0.85)
            base_dir: Optional base directory for database files (default: ~/.comedybot)
        """
        self.similarity_threshold = similarity_threshold
        self.dimension = 300  # spaCy vector dimension
        
        # Setup directory structure
        self.base_dir = os.path.expanduser(base_dir if base_dir else "~/.comedybot")
        self.vectors_dir = os.path.join(self.base_dir, "bit_vectors")
        self.registry_file = os.path.join(self.base_dir, "bit_registry.json")
        self.index_file = os.path.join(self.base_dir, "faiss_index.bin")
        
        # Create directories if they don't exist
        os.makedirs(self.vectors_dir, exist_ok=True)
        
        # Initialize or load registry
        self.registry = self._load_registry()
        
        # Initialize or load FAISS index
        self.index = self._load_index()
        
    def _load_registry(self) -> Dict[str, List[str]]:
        """Load the bit registry from disk or create if it doesn't exist."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_registry(self) -> None:
        """Save the current registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    def _load_index(self) -> faiss.Index:
        """Load the FAISS index from disk or create a new one."""
        if os.path.exists(self.index_file):
            return faiss.read_index(self.index_file)
        
        # Create new index
        index = faiss.IndexFlatL2(self.dimension)
        return index
        
    def _save_index(self) -> None:
        """Save the current FAISS index to disk."""
        faiss.write_index(self.index, self.index_file)
        
    def _save_bit_vector(self, bit_data: Dict[str, Any]) -> None:
        """Save a bit vector JSON file to the vectors directory."""
        hash_value = bit_data['bit_info']['term_vector_hash']
        vector_file = os.path.join(self.vectors_dir, f"{hash_value}.json")
        with open(vector_file, 'w') as f:
            json.dump(bit_data, f, indent=2)
            
    def _load_bit_vector(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """Load a bit vector JSON file from the vectors directory."""
        vector_file = os.path.join(self.vectors_dir, f"{hash_value}.json")
        if os.path.exists(vector_file):
            with open(vector_file, 'r') as f:
                return json.load(f)
        return None
        
    def _find_title_by_hash(self, hash_value: str) -> Optional[str]:
        """Find a bit title by its vector hash in the registry."""
        for title, hashes in self.registry.items():
            if hash_value in hashes:
                return title
        return None
        
    def _find_similar_bits(self, vector: List[float]) -> Optional[str]:
        """
        Find similar bits using FAISS index.
        Returns the title of the most similar bit if similarity exceeds threshold.
        """
        if self.index.ntotal == 0:
            return None
            
        # Convert vector to numpy array
        query_vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        
        # Search index
        D, I = self.index.search(query_vector, 1)
        distance = D[0][0]
        
        # Convert L2 distance to similarity (closer to 1 means more similar)
        similarity = 1 / (1 + distance)
        
        if similarity >= self.similarity_threshold:
            # Find which bit this vector belongs to
            for title, hashes in self.registry.items():
                for hash_value in hashes:
                    bit_data = self._load_bit_vector(hash_value)
                    if bit_data is None:
                        continue
                    
                    stored_vector = np.array(bit_data['bit_vector'], dtype=np.float32)
                    if np.array_equal(stored_vector, query_vector[0]):
                        return title
                        
        return None
        
    def compare(self, bit_data: Dict[str, Any]) -> str:
        """
        Compare a bit with the database and return its title.
        
        This method:
        1. Checks if the bit's hash exists in registry
        2. If not, uses FAISS to find similar bits
        3. If no similar bits found, adds as new bit
        
        Args:
            bit_data: Dictionary containing bit vector and metadata
            
        Returns:
            Title of the bit (either existing or new)
        """
        hash_value = bit_data['bit_info']['term_vector_hash']
        
        # Check if hash exists in registry
        title = self._find_title_by_hash(hash_value)
        if title:
            logger.info(f"Found existing bit: {title}")
            return title
            
        # Look for similar bits
        similar_title = self._find_similar_bits(bit_data['bit_vector'])
        if similar_title:
            logger.info(f"Found similar bit: {similar_title}")
            
            # Add new hash to existing title
            self.registry[similar_title].append(hash_value)
            self._save_registry()
            
            # Save bit vector
            self._save_bit_vector(bit_data)
            
            # Update index
            vector = np.array(bit_data['bit_vector'], dtype=np.float32).reshape(1, -1)
            self.index.add(vector)
            self._save_index()
            
            return similar_title
            
        # No similar bits found, add as new
        new_title = bit_data['bit_info']['title']
        logger.info(f"Adding new bit: {new_title}")
        
        # Add to registry
        self.registry[new_title] = [hash_value]
        self._save_registry()
        
        # Save bit vector
        self._save_bit_vector(bit_data)
        
        # Update index
        vector = np.array(bit_data['bit_vector'], dtype=np.float32).reshape(1, -1)
        self.index.add(vector)
        self._save_index()
        
        return new_title
