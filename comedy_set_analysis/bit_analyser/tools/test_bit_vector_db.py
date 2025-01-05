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
            ngram_vectors=[("ng", np.array([7,8,9], dtype=np.float32))],
            punchline_vectors=[("punch", np.array([10,11,12], dtype=np.float32), 1.0)]
        )
        self.storage.save_bit_vectors(bit_id, vectors)
        loaded = self.storage.load_bit_vectors(bit_id)
        self.assertIsNotNone(loaded)
        self.assertIn("full_vector", loaded)
        self.assertTrue((loaded["full_vector"] == np.array([1,2,3], dtype=np.float32)).all())

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
        self.test_dir = tempfile.mkdtemp(prefix='bitstorage_test_')
        # Patch the base directory so it doesn't collide with real data
        with patch.object(BitStorageManager, '__init__', return_value=None):
            # We'll manually set up the storage manager
            self.mock_storage = MagicMock(spec=BitStorageManager)
            self.mock_storage.base_dir = self.test_dir
            self.mock_storage.vectors_dir = os.path.join(self.test_dir, 'vectors')
            self.mock_storage.indices_dir = os.path.join(self.test_dir, 'indices')
            self.mock_storage.bits_dir = os.path.join(self.test_dir, 'bits')
            self.mock_storage.registry_file = os.path.join(self.test_dir, 'bit_registry.json')
            self.mock_storage.canonical_bits_file = os.path.join(self.test_dir, 'canonical_bits.json')
            os.makedirs(self.mock_storage.vectors_dir, exist_ok=True)
            os.makedirs(self.mock_storage.indices_dir, exist_ok=True)
            os.makedirs(self.mock_storage.bits_dir, exist_ok=True)

        # We create a real instance of BitVectorDB, but patching out the storage manager's constructor
        self.db = BitVectorDB(dimension=384, similarity_threshold=0.7)
        self.db.storage = self.mock_storage
        self.db.registry = {}

        # Replace indexes with small dimension indexes for testing
        import faiss
        self.db.full_index = faiss.IndexFlatL2(3)
        self.db.sentence_index = faiss.IndexFlatL2(3)
        self.db.ngram_index = faiss.IndexFlatL2(3)
        self.db.punchline_index = faiss.IndexFlatL2(3)

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
            full_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sentence_vectors=[],
            ngram_vectors=[],
            punchline_vectors=[]
        )

        returned_id = self.db.add_to_database(mock_entity, vectors)
        self.assertEqual(returned_id, "bit_abc")
        self.assertIn("bit_abc", self.db.registry)

    def test_find_matching_bits(self):
        # Add a single vector to full_index for demonstration
        sample_vec = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        self.db.full_index.add(sample_vec)

        # Put bit_xyz in the registry
        self.db.registry = {"bit_xyz": "2025-01-01T12:00:00"}

        # Also add bit_xyz to canonical bits to avoid RuntimeError
        # For instance, let's give it a "Sample Title"
        self.db.canonical_bits.add_bit("Sample Title", "bit_xyz")

        # Create a query with no sentences, ngrams, or punchlines
        query_vecs = BitVectors(
            full_vector=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            sentence_vectors=[], 
            ngram_vectors=[], 
            punchline_vectors=[]
        )

        matches = self.db.find_matching_bits(query_vecs)
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].bit_id, "bit_xyz")
        self.assertAlmostEqual(matches[0].overall_score, 0.5, delta=0.05)

    def test_multi_candidate_search(self):
        # 3 bits, each with different vectors
        vec_a = np.random.rand(384).astype(np.float32)
        vec_b = np.random.rand(384).astype(np.float32)
        vec_c = np.random.rand(384).astype(np.float32)
        
        # Insert into DB with IDs: "bit_a", "bit_b", "bit_c"
        for vec, bit_id in [(vec_a, "bit_a"), (vec_b, "bit_b"), (vec_c, "bit_c")]:
            vectors = BitVectors(
                full_vector=vec,
                sentence_vectors=[],
                ngram_vectors=[],
                punchline_vectors=[]
            )
            bit_entity = BitEntity(
                bit_id=bit_id,
                title=f"Test Bit {bit_id}",
                text="Test bit text",
                show_info={},
                joke_types=[],
                themes=[]
            )
            self.db.add_to_database(bit_entity, vectors)
        
        # Query with something close to vec_a
        query_vec = vec_a + np.random.normal(0, 0.0001, (384,)).astype(np.float32)
        query = BitVectors(full_vector=query_vec, sentence_vectors=[], ngram_vectors=[], punchline_vectors=[])
        
        # Perform search
        matches = self.db.find_matching_bits(query)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].bit_id, "bit_a")  # Expect "bit_a" to be top



if __name__ == '__main__':
    unittest.main()