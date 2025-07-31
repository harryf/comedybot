# Comedy Bit Analyser

A comprehensive toolkit for analyzing, comparing, and managing comedy bits using advanced natural language processing and vector similarity techniques.

## Overview

The Comedy Bit Analyser is designed to help comedians and comedy writers analyze their material, identify similarities between bits, track the evolution of jokes over time, and organize their comedy material efficiently. The system uses multi-level vector comparison techniques to identify similar comedy bits, even when they are worded differently but convey similar concepts or jokes.

## Core Components

### 1. Bit Comparison Tool

The `bit_comparison_tool.py` is the main entry point for comparing comedy bits. It analyzes comedy bits from a specified directory and compares them against a database of existing bits to identify similarities.

**Input:**
- Directory containing:
  - `bits.json`: Contains bit definitions and metadata
  - `metadata.json`: Contains show metadata and settings
  - `transcript_clean.json`: Contains processed transcript text

**Output:**
- Console output showing exact and potential matches
- Bits are added to the central database (~/.comedybot/)
- Vector representations are stored for future comparisons

**Usage:**
```bash
python bit_comparison_tool.py -d /path/to/bit/directory -t 0.7
```
Where:
- `-d` specifies the directory containing the input files
- `-t` specifies the similarity threshold (default: 0.7)

### 2. Vector Database (BitVectorDB)

The `bit_vector_db.py` manages the storage and retrieval of bit vectors, providing a searchable database of comedy bits.

**Key Features:**
- Multi-level comparison using FAISS indices:
  - Full bit vector comparison
  - Sentence-level matching
  - N-gram and punchline detection
- Persistent storage of vectors and indices
- Canonical bit management (grouping similar versions of the same bit)

**Storage Structure:**
- `~/.comedybot/vectors/`: Stores vector representations of bits
- `~/.comedybot/indices/`: Stores FAISS indices for fast similarity search
- `~/.comedybot/bits/`: Stores bit entities (full bit data)
- `~/.comedybot/bit_registry.json`: Registry of all bits in the database
- `~/.comedybot/canonical_bits.json`: Mapping of canonical bit names to bit IDs

### 3. Term Vector Tool

The `term_vector_tool.py` generates vector representations of comedy bits using sentence transformers and NLP techniques.

**Processing Pipeline:**
1. Text segmentation into sentences
2. Generation of sentence embeddings using Sentence-BERT
3. Extraction and weighting of n-grams
4. Identification of potential punchlines
5. Creation of multi-level vector representations

**Output:**
- `BitVectors` object containing:
  - Full vector representation of the entire bit
  - Individual sentence vectors
  - N-gram vectors with position information
  - Punchline vectors with importance weights

### 4. Bit Entity

The `bit_entity.py` defines the data structure for comedy bits, handling loading, saving, and manipulation of bit data.

**Structure:**
- Show information (venue, date, etc.)
- Bit information (title, themes, joke types)
- Transcript data (text, timing information)
- Metadata (processing information, timestamps)

## Data Flow

1. **Input Processing:**
   - A directory containing bit data is provided to `bit_comparison_tool.py`
   - Required files (`bits.json`, `metadata.json`, `transcript_clean.json`) are validated
   - Bit entities are created from the input data

2. **Vector Generation:**
   - `TermVectorTool` processes each bit's text to generate vector representations
   - Multiple levels of vectors are created (full bit, sentences, n-grams, punchlines)

3. **Similarity Comparison:**
   - `BitVectorDB` compares the new bit vectors against existing bits in the database
   - Multi-level comparison scores are calculated and weighted
   - Matches are classified as exact (score > 0.7) or potential (lower scores)

4. **Database Storage:**
   - New bits are added to the database with their vector representations
   - If a strong match is found, the bit is linked to an existing canonical bit
   - Otherwise, a new canonical bit entry is created
   - Vectors and indices are saved to the central storage location

5. **Results Reporting:**
   - A summary table shows exact and potential matches with their scores
   - Detailed information about matching n-grams and their positions is provided

## Advanced Features

### Canonical Bit Management

The system tracks different versions of the same bit across performances using a canonical bit system:

- Similar bits are grouped under a canonical title
- Each version maintains its unique ID and metadata
- Relationships between versions are preserved

### Joke Type and Theme Tracking

The system can categorize bits by joke types and themes:

- `joke_type_tool.py` identifies the types of jokes used
- `theme_identifier_tool.py` extracts thematic elements
- These categorizations are stored and can be used for filtering and analysis

### Vector Storage Optimization

The system uses a centralized storage approach:

- All vectors are stored in `~/.comedybot/vectors/`
- All indices are stored in `~/.comedybot/indices/`
- This ensures consistent access and prevents duplication

## Technical Details

### Vector Dimensions and Precision

- Default vector dimension: 384 (using all-MiniLM-L6-v2 model)
- Vector precision: 6 decimal places
- Vectors are normalized using L2 normalization

### Similarity Thresholds

- Hard match threshold: 0.7 (considered the same bit)
- Component weights:
  - Full vector comparison: 45%
  - Sentence matches: 30%
  - N-gram matches: 15%
  - Punchline matches: 10%

### Dependencies

- FAISS for efficient similarity search
- Sentence-BERT for text embeddings
- spaCy for NLP processing
- NumPy for vector operations
- Pydantic for data validation

## Troubleshooting

### Common Issues

1. **Empty Indices or Vectors Directories:**
   - Ensure that the bit_vector_db._save_indices() and _save_bit_vectors() methods are being called
   - Check permissions on the ~/.comedybot/ directory

2. **Missing Required Files:**
   - Ensure that bits.json, metadata.json, and transcript_clean.json exist in the input directory
   - Verify file permissions allow reading

3. **Vector Dimension Mismatches:**
   - If you change the embedding model, existing vectors may have different dimensions
   - Use the regenerate=True flag with TermVectorTool to rebuild vectors

## Current Limitations and Future Enhancements

### Current Limitations

1. **Surface Noise Hides Core Gags:**
   - Different riffs and tags around the same core joke can swamp the global vector
   - Example: Multiple versions of a "Fuchs/Fox/Fucks" surname joke appear as different bits due to surrounding content

2. **Brittle Sentence-Level Matching:**
   - Small wording changes ("family reunion" vs "family get-together") defeat cosine similarity
   - Current n-gram approach doesn't capture semantic equivalence well

3. **Outdated Embedding Model:**
   - The MiniLM model (2020) lags behind modern instruction-tuned models by 8-12 points on similarity tasks

### Proposed Enhancements

#### 1. Improved Embedding Models

- Replace MiniLM with stronger open-source alternatives:
  - **BGE-M3** or **BGE-base-en** (current MTEB leaderboard top performers)
  - **E5-mistral-7b-instruct** or **Jina Embeddings v2**
- Use instruction-tuning with prompts like "Represent the meaning of this stand-up bit for clustering similar jokes..."

#### 2. Advanced Vector Storage

- Implement hybrid search (dense vectors + BM25) using **Qdrant** or **Weaviate**
- Add payload filtering for metadata
- Consider **Milvus 3.2** for scalar filters

#### 3. LLM-Assisted Normalization

- Use LLMs (GPT-4o or Mixtral) to extract structured summaries of bits:
  ```json
  {
    "premise": "audience mispronounces my German surname 'Fuchs' as 'Fox/Fucks'",
    "comic_device": "wordplay",
    "core_subject": "pronunciation of surname",
    "signature_lines": ["key phrase 1", "key phrase 2"]
  }
  ```
- Embed these structured summaries instead of raw transcripts
- Store both summary vectors and full-text vectors for different query types

#### 4. Phonetic Processing

- Apply Metaphone or Cologne-phonetics to normalize names and words before embedding
- Example: "Fuchs", "Fooks", "Fux", "Fox" all collapse to /FKS/
- Makes n-gram matching more robust to pronunciation variations

#### 5. Reranking and Classification

- Add a reranker step (e.g., **bge-reranker-large** or **colbert-v2**) to improve results
- Optionally fine-tune a small classifier with 15+ examples per canonical bit
- Use supervised fine-tuning or LoRA heads for direct "same bit or not" classification

#### 6. Implementation Pipeline

```python
# Pseudo-pipeline
for bit in new_bits:
    summary_json = llm_tag(bit.transcript)
    bit.summary = json.dumps(summary_json, ensure_ascii=False)
    bit.embedding_summary = embed(summary_json)
    bit.embedding_full = embed(bit.transcript)

db.upsert(bit_id, {
    "sum_vec": bit.embedding_summary,
    "full_vec": bit.embedding_full,
    "date": bit.show_date,
    "device": summary_json["comic_device"],
    # ...
})
```

#### 7. Quick Wins

- Replace MiniLM with BGE-base-en (one-line change in sentence-transformers)
- Generate 30-word summaries for sample bits to test improved clustering
- Add hybrid search for lexical hooks
- Evaluate a reranker on top-20 results to reduce false positives
