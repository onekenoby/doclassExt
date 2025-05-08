
# README.md

# Document-to-Neo4j Graph System

This project allows you to extract text from documents (PDF, DOCX, or image files), understand them using Google Gemini, and build a knowledge graph in Neo4j.

## Setup

1. Clone the repo
2. Create a `.env` file in the root directory with the following content:

```
GEMINI_API_KEY=your_gemini_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python main.py
```

## Features

- PDF/DOCX/image parsing
- OCR fallback for scanned documents
- Gemini VLM-to-Cypher generation
- Graph creation in Neo4j

## Sample files

Place your test files inside the `samples/` directory.
