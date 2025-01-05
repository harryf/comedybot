#!/usr/bin/env python3

from bit_vector_db import BitVectorDB
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def print_comparison_result(name: str, vector1: list, vector2: list, db: BitVectorDB):
    similarity = db.debug_compare_vectors(vector1, vector2)
    print(f"\nTest: {name}")
    print(f"Cosine similarity: {similarity}")
    print(f"Threshold: {db.similarity_threshold}")
    print(f"Match? {'Yes' if similarity >= db.similarity_threshold else 'No'}")
    print("-" * 50)

def main():
    # Initialize BitVectorDB with default threshold
    db = BitVectorDB()
    
    # Original vectors (very similar bits)
    original_vector1 = [-0.6844772100448608, 0.21998053789138794, -0.20061205327510834, -0.09617000818252563, -0.12680457532405853, 0.01876785047352314, 0.09781669080257416, -0.3223396837711334, 0.01755865477025509, 1.9644688367843628, -0.1962212324142456, -0.10369066148996353, -0.057515401393175125, 0.010661648586392403, -0.3014950454235077, -0.06452590972185135, -0.04520730301737785, 0.7232325077056885, -0.11518567055463791, -0.018101517111063004]  # truncated for readability
    
    original_vector2 = [-0.6756075620651245, 0.17080914974212646, -0.22476713359355927, -0.11075953394174576, -0.12381429225206375, 0.022595008835196495, 0.08932683616876602, -0.2998051345348358, 0.012019449844956398, 1.9663292169570923, -0.17426587641239166, -0.08240004628896713, -0.048662640154361725, 0.017320478335022926, -0.2684842050075531, -0.0406070314347744, -0.04439210891723633, 0.6097027063369751, -0.12567317485809326, -0.00414591608569026]  # truncated for readability
    
    # Slightly modified vector (similar pattern but with some variations)
    modified_vector = [-0.67, 0.19, -0.21, -0.10, -0.12, 0.02, 0.09, -0.30, 0.015, 1.96, -0.18, -0.09, -0.05, 0.015, -0.28, -0.05, -0.04, 0.65, -0.12, -0.01]
    
    # Completely different vector (different pattern entirely)
    different_vector = [0.5, -0.3, 0.4, 0.2, 0.1, -0.4, 0.3, 0.6, -0.2, -1.5, 0.4, 0.3, 0.1, -0.2, 0.5, 0.2, 0.1, -0.5, 0.3, 0.2]
    
    # Run tests
    print("\nRunning similarity tests with threshold:", db.similarity_threshold)
    print("=" * 50)
    
    # Test 1: Original similar bits
    print_comparison_result(
        "Original Similar Bits",
        original_vector1,
        original_vector2,
        db
    )
    
    # Test 2: Original vs Modified
    print_comparison_result(
        "Original vs Modified Version",
        original_vector1,
        modified_vector,
        db
    )
    
    # Test 3: Original vs Different
    print_comparison_result(
        "Original vs Different Bit",
        original_vector1,
        different_vector,
        db
    )

if __name__ == "__main__":
    main()
