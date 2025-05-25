import os
import sys

# Ensure the script can access parent directory for 'docling' package and project-level dirs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_SCRIPT_DIR)) # Adds 'knowledge' directory to sys.path
from typing import List

def main():
    import lancedb
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import EmbeddingFunctionRegistry

    # --- Project Root Definition ---
    # Assuming this script is in PROJECT_ROOT/knowledge/docling/
    _PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

    # --- Configuration ---
    # MODIFIED LINE: PDF_DIR now points to a 'pdfs' folder within the same directory as this script
    PDF_DIR = os.path.join(_SCRIPT_DIR, "pdfs") 
    DB_PATH = os.path.join(_PROJECT_ROOT, "data", "lancedb") # DB remains in project_root/data/lancedb
    TABLE_NAME = "emiAI_docs"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    TOKENIZER_NAME = 'sentence-transformers/all-MiniLM-L6-v2' 
    CHUNK_MAX_TOKENS = 256 
    EMBEDDING_DIMENSION = 384

    # Ensure DB directory exists
    print(f"Ensuring database directory exists at: {DB_PATH}")
    os.makedirs(DB_PATH, exist_ok=True)

    # --- LanceDB Schema Definition ---
    class ChunkMetadata(LanceModel):
        """Metadata for each chunk."""
        filename: str
        page_numbers: List[int] | None
        title: str | None

    class ChunksTable(LanceModel):
        """Schema for the LanceDB table."""
        text: str
        vector: Vector(EMBEDDING_DIMENSION) 
        metadata: ChunkMetadata

    def preprocess_and_embed_pdfs():
        """
        Processes all PDFs in the PDF_DIR, chunks them, generates embeddings,
        and stores them in a LanceDB table.
        """
        print(f"Connecting to LanceDB at: {DB_PATH}")
        db = lancedb.connect(DB_PATH)
        
        print(f"Attempting to create/overwrite table: {TABLE_NAME}")
        try:
            table = db.create_table(TABLE_NAME, schema=ChunksTable, mode="overwrite")
        except Exception as e:
            print(f"Error creating table: {e}")
            print("Attempting to open table if it already exists...")
            table = db.open_table(TABLE_NAME)

        print("Initializing models...")
        converter = DocumentConverter()
        # Load the sentence transformer model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Load the tokenizer for the HybridChunker
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        all_processed_chunks_for_db = []
        
        print(f"Scanning PDF directory: {PDF_DIR}")
        if not os.path.exists(PDF_DIR):
            print(f"Error: PDF directory '{PDF_DIR}' not found.")
            return
        if not os.listdir(PDF_DIR):
            print(f"No PDF files found in '{PDF_DIR}'.")
            return

        for filename in os.listdir(PDF_DIR):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(PDF_DIR, filename)
                print(f"\nProcessing PDF: {file_path}")

                try:
                    # 1. Extract content using DocumentConverter
                    print(f"  Extracting content from {filename}...")
                    conversion_result = converter.convert(file_path)
                    if not conversion_result or not conversion_result.document:
                        print(f"  Could not convert or extract document from {filename}")
                        continue
                    
                    docling_document = conversion_result.document
                    print(f"  Extraction successful for {filename}.")

                    # 2. Apply hybrid chunking
                    print(f"  Chunking document {filename}...")
                    chunk_iter = chunker.chunk(dl_doc=docling_document)
                    doc_chunks = list(chunk_iter)
                    print(f"  Found {len(doc_chunks)} chunks for {filename}.")

                    if not doc_chunks:
                        print(f"  No chunks generated for {filename}.")
                        continue

                    # 3. Prepare chunks for the table (including embedding)
                    print(f"  Preparing and embedding chunks for {filename}...")
                    for i, chunk in enumerate(doc_chunks):
                        text_content = chunk.text
                        if not text_content or not text_content.strip():
                            print(f"    Skipping empty chunk {i+1} from {filename}.")
                            continue

                        # Generate embedding for the chunk text
                        vector = embedding_model.encode(text_content).tolist()

                        page_nos = sorted(
                            list(
                                set(
                                    prov.page_no
                                    for item in chunk.meta.doc_items
                                    for prov in item.prov
                                    if prov.page_no is not None
                                )
                            )
                        ) or None

                        chunk_data_for_db = {
                            "text": text_content,
                            "vector": vector,
                            "metadata": {
                                "filename": filename,
                                "page_numbers": page_nos,
                                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                            }
                        }
                        all_processed_chunks_for_db.append(chunk_data_for_db)
                    print(f"  Finished preparing {len(doc_chunks)} chunks from {filename}.")

                except Exception as e:
                    print(f"  Error processing file {filename}: {e}")
                    import traceback
                    traceback.print_exc()

        if not all_processed_chunks_for_db:
            print("\nNo chunks were processed from any PDF. Database will not be updated.")
            return

        # 4. Add all collected chunks to the LanceDB table
        print(f"\nAdding {len(all_processed_chunks_for_db)} processed chunks to LanceDB table '{TABLE_NAME}'...")
        try:
            table.add(all_processed_chunks_for_db)
            print("Successfully added chunks to the database.")
            print(f"Total rows in table: {table.count_rows()}")
        except Exception as e:
            print(f"Error adding data to LanceDB: {e}")
            import traceback
            traceback.print_exc()

    # Run the preprocessing
    preprocess_and_embed_pdfs()
    print("\nPreprocessing script finished.")

if __name__ == "__main__":
    main()