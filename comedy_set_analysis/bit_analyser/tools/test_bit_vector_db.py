import os
import sys

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import shutil
import tempfile
import uuid
import json
import numpy as np

from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional

from bit_vector_db import *

class TestBitStorageManager(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory in /tmp to test filesystem operations
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        self.storage = BitStorageManager(base_dir=self.test_dir)

    def tearDown(self):
        # Cleanup: remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_directories(self):
        """Test that the storage manager creates all directories."""
        self.assertTrue(os.path.exists(self.storage.base_dir))
        self.assertTrue(os.path.exists(self.storage.vectors_dir))
        self.assertTrue(os.path.exists(self.storage.indices_dir))
        self.assertTrue(os.path.exists(self.storage.bits_dir))

    def test_save_and_load_registry(self):
        """Test saving and loading the registry file."""
        reg = {"bit123": "2025-01-01T12:00:00"}
        self.storage.save_registry(reg)
        loaded = self.storage.load_registry()
        self.assertIn("bit123", loaded)
        self.assertEqual(loaded["bit123"], "2025-01-01T12:00:00")

    def test_save_bit_vectors_and_load(self):
        """Test saving and loading a bit's vectors."""
        bit_id = "bit_abc"
        vectors = BitVectors(
            full_vector=np.array([1,2,3], dtype=np.float32),
            sentence_vectors=[np.array([4,5,6], dtype=np.float32)],
            ngram_vectors=[("ng", np.array([7,8,9], dtype=np.float32), 0)],
            punchline_vectors=[("punch", np.array([10,11,12], dtype=np.float32), 1.0)]
        )
        self.storage.save_bit_vectors(bit_id, vectors)
        loaded = self.storage.load_bit_vectors(bit_id)
        self.assertIsNotNone(loaded)
        self.assertTrue(isinstance(loaded, BitVectors))
        self.assertTrue(np.array_equal(loaded.full_vector, np.array([1,2,3], dtype=np.float32)))
        self.assertTrue(np.array_equal(loaded.sentence_vectors[0], np.array([4,5,6], dtype=np.float32)))
        self.assertEqual(len(loaded.ngram_vectors), 1)
        self.assertEqual(loaded.ngram_vectors[0][0], "ng")
        self.assertTrue(np.array_equal(loaded.ngram_vectors[0][1], np.array([7,8,9], dtype=np.float32)))
        self.assertEqual(loaded.ngram_vectors[0][2], 0)
        self.assertEqual(len(loaded.punchline_vectors), 1)
        self.assertEqual(loaded.punchline_vectors[0][0], "punch")
        self.assertTrue(np.array_equal(loaded.punchline_vectors[0][1], np.array([10,11,12], dtype=np.float32)))
        self.assertEqual(loaded.punchline_vectors[0][2], 1.0)

    def test_delete_bit_data(self):
        """Test deleting a bit's data file."""
        bit_id = "bit_to_delete"
        vectors = BitVectors(
            full_vector=np.array([1,1,1], dtype=np.float32),
            sentence_vectors=[], ngram_vectors=[], punchline_vectors=[]
        )
        self.storage.save_bit_vectors(bit_id, vectors)
        path = os.path.join(self.storage.vectors_dir, bit_id + ".npz")
        self.assertTrue(os.path.exists(path))
        self.storage.delete_bit_data(bit_id)
        self.assertFalse(os.path.exists(path))

    def test_canonical_bits_save_and_load(self):
        """Test saving and loading canonical bits."""
        self.storage.save_canonical_bits({"myBit": ["bit123"]})
        loaded = self.storage.load_canonical_bits()
        self.assertIn("myBit", loaded)
        self.assertEqual(loaded["myBit"], ["bit123"])

    def test_save_and_load_data(self):
        """Test saving and loading arbitrary data (themes, joke_types, etc.)."""
        self.storage.save_data("test_data.json", {"key": "value"})
        loaded = self.storage.load_data("test_data.json")
        self.assertEqual(loaded.get("key"), "value")


class TestCanonicalBits(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        self.storage = BitStorageManager(base_dir=self.test_dir)
        self.cbits = CanonicalBits(self.storage)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_bit_and_retrieve_by_title(self):
        self.cbits.add_bit("My New Bit", "bit01")
        result = self.cbits.get_bit_by_title("My New Bit")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "My New Bit")
        self.assertIn("bit01", result[1])

    def test_add_bit_with_matching_bit_id(self):
        # Suppose bit02 is in the same group as bit01
        self.cbits.add_bit("My Existing Bit", "bit01")
        self.cbits.add_bit("Another Title", "bit02", matching_bit_id="bit01")
        # Now bit02 should be in the same group as bit01
        self.assertIsNotNone(self.cbits.get_bit_by_id("bit02"))
        title, bit_ids = self.cbits.get_bit_by_id("bit02")
        self.assertIn("bit02", bit_ids)
        # The original title for "bit01" should still exist
        self.assertIsNotNone(self.cbits.get_bit_by_title("My Existing Bit"))


class TestJokeTypeTracker(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        self.storage = BitStorageManager(base_dir=self.test_dir)
        self.tracker = JokeTypeTracker(self.storage)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_bit(self):
        self.tracker.add_bit("bit101", ["pun", "dark"])
        self.assertIn("pun", self.tracker.joke_type_map)
        self.assertIn("bit101", self.tracker.joke_type_map["pun"])
        self.assertIn("dark", self.tracker.joke_type_map)
        self.assertIn("bit101", self.tracker.joke_type_map["dark"])


class TestThemeTracker(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        self.storage = BitStorageManager(base_dir=self.test_dir)
        self.tracker = ThemeTracker(self.storage)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_bit(self):
        self.tracker.add_bit("bit999", ["relationships", "parenting"])
        self.assertIn("relationships", self.tracker.theme_map)
        self.assertIn("bit999", self.tracker.theme_map["relationships"])
        self.assertIn("parenting", self.tracker.theme_map)
        self.assertIn("bit999", self.tracker.theme_map["parenting"])


class TestBitVectorDB(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        
        # Create a real BitStorageManager with the test directory
        self.storage = BitStorageManager(base_dir=self.test_dir)
        
        # Create a real instance of BitVectorDB
        self.db = BitVectorDB(dimension=384, similarity_threshold=0.7)
        
        # Replace the storage manager with our test one
        self.db.storage = self.storage
        self.db.registry = {}

        # Replace indexes with small dimension indexes for testing
        import faiss
        self.db.full_index = faiss.IndexFlatL2(384)
        self.db.sentence_index = faiss.IndexFlatL2(384)
        self.db.ngram_index = faiss.IndexFlatL2(384)
        self.db.punchline_index = faiss.IndexFlatL2(384)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_has_bit(self):
        self.db.registry = {"bit123": "2025-01-01T12:00:00"}
        self.assertTrue(self.db.has_bit("bit123"))
        self.assertFalse(self.db.has_bit("bit_unknown"))

    def test_add_to_database(self):
        # mock a BitEntity
        mock_entity = MagicMock(spec=BitEntity)
        mock_entity.bit_data = {
            "bit_id": "bit_abc",
            "bit_info": {"title": "My Mock Bit"},
            "show_info": {}
        }

        # create some vectors
        vectors = BitVectors(
            full_vector=np.random.rand(384).astype(np.float32),
            sentence_vectors=[],
            ngram_vectors=[],
            punchline_vectors=[]
        )

        returned_id = self.db.add_to_database(mock_entity, vectors)
        self.assertEqual(returned_id, "bit_abc")
        self.assertIn("bit_abc", self.db.registry)

    def test_find_matching_bits(self):
        # Create a normalized vector for testing
        sample_vec = np.ones((1, 384), dtype=np.float32)
        sample_vec = sample_vec / np.linalg.norm(sample_vec)  # Normalize to unit vector
        faiss.normalize_L2(sample_vec)  # FAISS normalization

        # Add to database
        vectors = BitVectors(
            full_vector=sample_vec[0],
            sentence_vectors=[],
            ngram_vectors=[],
            punchline_vectors=[]
        )
        bit_entity = BitEntity("dummy_path")
        bit_entity.bit_data = {
            "bit_id": "bit_xyz",
            "show_info": {},
            "bit_info": {
                "title": "Sample Title",
                "joke_types": [],
                "themes": []
            },
            "transcript": {
                "text": "Test bit text",
            },
            "audience_reactions": []
        }
        self.db.add_to_database(bit_entity, vectors)

        # Also add bit_xyz to canonical bits
        self.db.canonical_bits.add_bit("Sample Title", "bit_xyz")

        # Create an almost identical query vector (small perturbation)
        query_vec = sample_vec[0] + np.random.normal(0, 0.01, (384,)).astype(np.float32)
        query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)  # FAISS normalization
        
        query_vecs = BitVectors(
            full_vector=query_vec[0],
            sentence_vectors=[], 
            ngram_vectors=[], 
            punchline_vectors=[]
        )

        matches = self.db.find_matching_bits(query_vecs)
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].bit_id, "bit_xyz")
        self.assertGreater(matches[0].overall_score, 0.7)  # Should be very similar

    def test_multi_candidate_search(self):
        # Create normalized base vectors
        base_vec = np.ones(384, dtype=np.float32)
        base_vec = base_vec.reshape(1, -1)
        faiss.normalize_L2(base_vec)  # FAISS normalization
        base_vec = base_vec[0]
        
        # Create 3 similar but distinct vectors with increasing perturbations
        vec_a = base_vec + np.random.normal(0, 0.01, (384,)).astype(np.float32)
        vec_a = vec_a.reshape(1, -1)
        faiss.normalize_L2(vec_a)
        vec_a = vec_a[0]
        
        vec_b = base_vec + np.random.normal(0, 0.1, (384,)).astype(np.float32)
        vec_b = vec_b.reshape(1, -1)
        faiss.normalize_L2(vec_b)
        vec_b = vec_b[0]
        
        vec_c = base_vec + np.random.normal(0, 0.2, (384,)).astype(np.float32)
        vec_c = vec_c.reshape(1, -1)
        faiss.normalize_L2(vec_c)
        vec_c = vec_c[0]
        
        # Insert into DB with IDs: "bit_a", "bit_b", "bit_c"
        for vec, bit_id in [(vec_a, "bit_a"), (vec_b, "bit_b"), (vec_c, "bit_c")]:
            vectors = BitVectors(
                full_vector=vec,
                sentence_vectors=[],
                ngram_vectors=[],
                punchline_vectors=[]
            )
            bit_entity = BitEntity("dummy_path")
            bit_entity.bit_data = {
                "bit_id": bit_id,
                "show_info": {},
                "bit_info": {
                    "title": f"Test Bit {bit_id}",
                    "joke_types": [],
                    "themes": []
                },
                "transcript": {
                    "text": "Test bit text",
                },
                "audience_reactions": []
            }
            self.db.add_to_database(bit_entity, vectors)
            self.db.canonical_bits.add_bit(f"Test Bit {bit_id}", bit_id)
        
        # Query with something very close to vec_a
        query_vec = vec_a + np.random.normal(0, 0.001, (384,)).astype(np.float32)
        query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        query = BitVectors(
            full_vector=query_vec[0],
            sentence_vectors=[],
            ngram_vectors=[],
            punchline_vectors=[]
        )
        
        # Perform search
        matches = self.db.find_matching_bits(query)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].bit_id, "bit_a")  # Expect "bit_a" to be top match

    def test_sentence_match_boost(self):
        """Test that strong sentence matches can boost overall ranking."""
        # Create a base vector that both bits will be similar to
        base_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)  # Unit vector
        
        # Create a sentence vector that will be shared between bit1 and query
        sentence_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)  # Unit vector

        # Bit 1: Similar full vector + matching sentence
        bit1_full = base_vec + np.random.normal(0, 0.01, (384,)).astype(np.float32)  # Small perturbation
        bit1_full = bit1_full / np.linalg.norm(bit1_full)  # Normalize
        bit1_vectors = BitVectors(
            full_vector=bit1_full,
            sentence_vectors=[sentence_vec],  # Exact matching sentence
            ngram_vectors=[],
            punchline_vectors=[]
        )
        bit1_entity = BitEntity("dummy_path")
        bit1_entity.bit_data = {
            "bit_id": "bit1",
            "show_info": {},
            "bit_info": {
                "title": "Bit 1",
                "joke_types": [],
                "themes": []
            },
            "transcript": {
                "text": "Test bit text with matching sentence",
            },
            "audience_reactions": []
        }
        self.db.add_to_database(bit1_entity, bit1_vectors)

        # Bit 2: Similar full vector but different sentence
        bit2_full = base_vec + np.random.normal(0, 0.01, (384,)).astype(np.float32)  # Small perturbation
        bit2_full = bit2_full / np.linalg.norm(bit2_full)  # Normalize
        bit2_sentence = np.random.rand(384).astype(np.float32)
        bit2_sentence = bit2_sentence / np.linalg.norm(bit2_sentence)  # Normalize
        bit2_vectors = BitVectors(
            full_vector=bit2_full,
            sentence_vectors=[bit2_sentence],
            ngram_vectors=[],
            punchline_vectors=[]
        )
        bit2_entity = BitEntity("dummy_path")
        bit2_entity.bit_data = {
            "bit_id": "bit2",
            "show_info": {},
            "bit_info": {
                "title": "Bit 2",
                "joke_types": [],
                "themes": []
            },
            "transcript": {
                "text": "Test bit text with different sentence",
            },
            "audience_reactions": []
        }
        self.db.add_to_database(bit2_entity, bit2_vectors)

        # Query: Similar to both in full vector, but has matching sentence with bit1
        query_full = base_vec + np.random.normal(0, 0.01, (384,)).astype(np.float32)  # Small perturbation
        query_full = query_full / np.linalg.norm(query_full)  # Normalize
        query_vectors = BitVectors(
            full_vector=query_full,
            sentence_vectors=[sentence_vec],  # Exact match with bit1's sentence
            ngram_vectors=[],
            punchline_vectors=[]
        )

        # Find matches
        matches = self.db.find_matching_bits(query_vectors)
        
        # Verify we got matches
        self.assertGreater(len(matches), 0)
        
        # Verify bit1 (with matching sentence) ranks higher than bit2
        bit1_rank = next((i for i, m in enumerate(matches) if m.bit_id == "bit1"), -1)
        bit2_rank = next((i for i, m in enumerate(matches) if m.bit_id == "bit2"), -1)
        
        self.assertNotEqual(bit1_rank, -1, "Bit1 should be in matches")
        self.assertNotEqual(bit2_rank, -1, "Bit2 should be in matches")
        self.assertLess(bit1_rank, bit2_rank, "Bit1 should rank higher than Bit2 due to sentence match")

    def test_ngram_match_boost(self):
        """Test that strong n-gram matches can boost overall ranking."""
        # Create normalized base vectors
        base_vec = np.ones(384, dtype=np.float32)
        base_vec = base_vec.reshape(1, -1)
        faiss.normalize_L2(base_vec)
        base_vec = base_vec[0]
        
        # Create bit vectors with controlled differences
        bit1_full = base_vec + np.random.normal(0, 0.1, (384,)).astype(np.float32)
        bit1_full = bit1_full.reshape(1, -1)
        faiss.normalize_L2(bit1_full)
        bit1_full = bit1_full[0]
        
        bit2_full = base_vec + np.random.normal(0, 0.05, (384,)).astype(np.float32)
        bit2_full = bit2_full.reshape(1, -1)
        faiss.normalize_L2(bit2_full)
        bit2_full = bit2_full[0]
        
        # Create identical ngram vectors for bit1 and query
        ngram_base = np.ones(384, dtype=np.float32)
        ngram_base = ngram_base.reshape(1, -1)
        faiss.normalize_L2(ngram_base)
        ngram_base = ngram_base[0]
        
        ngram_vec1 = ngram_base + np.random.normal(0, 0.001, (384,)).astype(np.float32)
        ngram_vec1 = ngram_vec1.reshape(1, -1)
        faiss.normalize_L2(ngram_vec1)
        ngram_vec1 = ngram_vec1[0]
        
        ngram_vec2 = ngram_base + np.random.normal(0, 0.001, (384,)).astype(np.float32)
        ngram_vec2 = ngram_vec2.reshape(1, -1)
        faiss.normalize_L2(ngram_vec2)
        ngram_vec2 = ngram_vec2[0]
        
        ngram_vec3 = ngram_base + np.random.normal(0, 0.001, (384,)).astype(np.float32)
        ngram_vec3 = ngram_vec3.reshape(1, -1)
        faiss.normalize_L2(ngram_vec3)
        ngram_vec3 = ngram_vec3[0]

        # Add bit1 with matching ngrams
        bit1_vectors = BitVectors(
            full_vector=bit1_full,
            sentence_vectors=[],
            ngram_vectors=[
                ("identical ngram 1", ngram_vec1, 0),
                ("identical ngram 2", ngram_vec2, 20),
                ("identical ngram 3", ngram_vec3, 40)
            ],
            punchline_vectors=[]
        )
        bit1_entity = BitEntity("dummy_path")
        bit1_entity.bit_data = {
            "bit_id": "bit1",
            "show_info": {},
            "bit_info": {
                "title": "Bit 1",
                "joke_types": [],
                "themes": []
            },
            "transcript": {"text": "Test bit text"},
            "audience_reactions": []
        }
        self.db.add_to_database(bit1_entity, bit1_vectors)
        self.db.canonical_bits.add_bit("Bit 1", "bit1")

        # Add bit2 with different ngrams
        different_ngram_vec1 = np.random.normal(0, 1, (384,)).astype(np.float32)
        different_ngram_vec1 = different_ngram_vec1.reshape(1, -1)
        faiss.normalize_L2(different_ngram_vec1)
        different_ngram_vec1 = different_ngram_vec1[0]
        
        different_ngram_vec2 = np.random.normal(0, 1, (384,)).astype(np.float32)
        different_ngram_vec2 = different_ngram_vec2.reshape(1, -1)
        faiss.normalize_L2(different_ngram_vec2)
        different_ngram_vec2 = different_ngram_vec2[0]
        
        different_ngram_vec3 = np.random.normal(0, 1, (384,)).astype(np.float32)
        different_ngram_vec3 = different_ngram_vec3.reshape(1, -1)
        faiss.normalize_L2(different_ngram_vec3)
        different_ngram_vec3 = different_ngram_vec3[0]

        bit2_vectors = BitVectors(
            full_vector=bit2_full,
            sentence_vectors=[],
            ngram_vectors=[
                ("different ngram 1", different_ngram_vec1, 0),
                ("different ngram 2", different_ngram_vec2, 100),
                ("different ngram 3", different_ngram_vec3, 200)
            ],
            punchline_vectors=[]
        )
        bit2_entity = BitEntity("dummy_path")
        bit2_entity.bit_data = {
            "bit_id": "bit2",
            "show_info": {},
            "bit_info": {
                "title": "Bit 2",
                "joke_types": [],
                "themes": []
            },
            "transcript": {"text": "Test bit text"},
            "audience_reactions": []
        }
        self.db.add_to_database(bit2_entity, bit2_vectors)
        self.db.canonical_bits.add_bit("Bit 2", "bit2")

        # Create query vectors with same ngrams as bit1
        query_full = base_vec + np.random.normal(0, 0.1, (384,)).astype(np.float32)
        query_full = query_full.reshape(1, -1)
        faiss.normalize_L2(query_full)
        query_full = query_full[0]
        
        query_vectors = BitVectors(
            full_vector=query_full,
            sentence_vectors=[],
            ngram_vectors=[
                ("identical ngram 1", ngram_vec1, 0),
                ("identical ngram 2", ngram_vec2, 20),
                ("identical ngram 3", ngram_vec3, 40)
            ],
            punchline_vectors=[]
        )

        # Find matches
        matches = self.db.find_matching_bits(query_vectors)
        
        # Verify we got matches
        self.assertGreater(len(matches), 0)
        
        # Verify bit1 (with matching n-grams) ranks higher than bit2
        bit1_rank = next((i for i, m in enumerate(matches) if m.bit_id == "bit1"), -1)
        bit2_rank = next((i for i, m in enumerate(matches) if m.bit_id == "bit2"), -1)
        
        self.assertNotEqual(bit1_rank, -1, "Bit1 should be in matches")
        self.assertNotEqual(bit2_rank, -1, "Bit2 should be in matches")
        self.assertLess(bit1_rank, bit2_rank, "Bit1 should rank higher than Bit2 due to n-gram matches")


if __name__ == '__main__':
    unittest.main()