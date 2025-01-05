#!/usr/bin/env python3

"""
Data structures for representing bit vectors.
"""

import numpy as np
from typing import List, Tuple
from pydantic import BaseModel, Field

class BitVectors(BaseModel):
    """
    Container for all vector representations of a bit.
    
    Attributes:
        full_vector: Vector representation of the entire bit text
        sentence_vectors: List of vectors for each sentence
        ngram_vectors: List of (text, vector) tuples for n-grams
        punchline_vectors: List of (text, vector, weight) tuples for punchlines
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    full_vector: np.ndarray = Field(description="Vector for full bit text")
    sentence_vectors: List[np.ndarray] = Field(description="Vectors for each sentence")
    ngram_vectors: List[Tuple[str, np.ndarray]] = Field(description="N-gram text and vectors")
    punchline_vectors: List[Tuple[str, np.ndarray, float]] = Field(description="Punchline text, vectors and weights")
