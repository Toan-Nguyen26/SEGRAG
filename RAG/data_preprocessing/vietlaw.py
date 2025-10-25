import json
import os
import re
import unicodedata
import nltk
from tqdm import tqdm

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt', quiet=True)


def sanitize_filename(filename):
    """Simple filename sanitization"""
    # Remove or replace characters that are problematic in filenames
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def create_concatenated_documents_viet_law_jsonl(
    txt_folder='data/vietnamese_law/chapters',
    qa_json_path='data/vietnamese_law/hanhchinh.json',
    output_dir='data/vietnamese_law',
    output_file='vietnamese_law.jsonl'
):
    """
    Process Vietnamese legal documents from a folder of .txt files into JSONL format.
    
    Each .txt file represents one chapter of the legal document.
    Creates N+1 entries: N chapter entries (with sentences, no Q&A) + 1 Q&A entry (with Q&A, no sentences)
    
    Args:
        txt_folder (str): Path to folder containing .txt files (one per chapter)
        qa_json_path (str): Path to JSON file containing Q&A data
        output_dir (str): Directory to save the output JSONL file
        output_file (str): Name of the output JSONL file
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output JSONL file
    out_file = os.path.join(output_dir, output_file)
    
    # Get all .txt files from the folder
    if not os.path.exists(txt_folder):
        print(f"❌ Error: Folder '{txt_folder}' does not exist!")
        return None
    
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    txt_files.sort()  # Sort alphabetically
    
    if not txt_files:
        print(f"❌ Error: No .txt files found in '{txt_folder}'")
        return None
    
    print(f"Found {len(txt_files)} .txt files in '{txt_folder}':")
    for f in txt_files:
        print(f"  - {f}")
    
    # Load Q&A data
    if not os.path.exists(qa_json_path):
        print(f"❌ Error: Q&A file '{qa_json_path}' does not exist!")
        return None
    
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"Loaded {len(qa_data)} Q&A pairs from '{qa_json_path}'")
    
    examples = []
    total_sentences = 0
    
    # Process each chapter (txt file)
    print("\nProcessing chapters...")
    for i, txt_file in enumerate(tqdm(txt_files, desc="Processing txt files")):
        txt_path = os.path.join(txt_folder, txt_file)
        
        # Read the text file - try UTF-8 variants first, avoid latin-1/cp1252 
        # which will misread UTF-8 bytes
        text = None
        tried_encodings = []
        
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be']:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text = f.read()
                if encoding != 'utf-8':
                    print(f"⚠️  {txt_file} read with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                tried_encodings.append(encoding)
                continue
        
        if text is None:
            print(f"❌ Error: Could not decode {txt_file} with UTF encodings: {tried_encodings}")
            print(f"   Your file may not be UTF-8. Please convert it using:")
            print(f"   python fix_encoding.py {txt_path}")
            continue
        
        # Normalize Unicode for Vietnamese (NFC is the standard)
        text = unicodedata.normalize('NFC', text)
        
        # PREPROCESSING FOR VIETNAMESE LEGAL DOCUMENTS
        
        # Step 1: Remove standalone numbered list markers (1., 2., 3., etc. on their own line)
        # Match: line break + number + dot + line break
        text = re.sub(r'\n(\d+)\.\s*\n', '\n', text)
        
        # Step 2: Also remove numbered markers at start of sentences but keep the content
        # This handles cases like "1. Luật này..." → "Luật này..."
        text = re.sub(r'(?:^|\n)(\d+)\.\s+', '\n', text, flags=re.MULTILINE)
        
        # Step 3: Ensure "Điều X." and its title are on the same line
        # Match: "Điều [number]." followed by line break and title
        # Example: "Điều 1.\nPhạm vi điều chỉnh" → "Điều 1. Phạm vi điều chỉnh"
        text = re.sub(r'(Điều\s+\d+\.)\s*\n\s*([^\n]+)', r'\1 \2', text)
        
        # DEBUG: Show first 200 chars to verify preprocessing worked
        print(f"    DEBUG - First 200 chars after preprocessing: {text[:200]}")
        
        # Minimal cleaning - normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize multiple blank lines to max 2
        text = re.sub(r'[ \t]+', ' ', text)            # Normalize spaces/tabs to single space
        text = text.strip()
        
        # Skip if empty
        if not text:
            print(f"⚠️  Skipping {txt_file} - empty file")
            continue
        
        # Skip if too long
        if len(text) > 100_000:
            print(f"⚠️  Skipping {txt_file} - too long ({len(text)} chars)")
            continue
        
        # Extract chapter title from first line or filename
        first_line = text.split('\n')[0].strip() if '\n' in text else text[:100]
        
        # Try to match Vietnamese chapter patterns
        chapter_pattern = r'(Chương\s+[IVX\d]+)\s*[-–—]?\s*([^\n]*)'
        match = re.search(chapter_pattern, first_line, re.IGNORECASE)
        
        if match:
            chapter_identifier = match.group(1).strip()
            chapter_title_part = match.group(2).strip()
            if chapter_title_part:
                title = f"{chapter_identifier} - {chapter_title_part}"
            else:
                title = chapter_identifier
        else:
            # Fallback: use filename as title
            title = os.path.splitext(txt_file)[0].replace('_', ' ').title()
        
        # Tokenize into sentences using NLTK
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"❌ Error tokenizing {txt_file}: {e}")
            continue
        
        # Clean sentences - just strip whitespace, don't modify content
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            print(f"⚠️  Skipping {txt_file} - no sentences after tokenization")
            continue
        
        # Create example dictionary (matching qasper/narrativeqa format)
        example = {
            "file": sanitize_filename(f"{i+1}_{title}.json"),
            "sentences": sentences,
            "labels": [],
            "title": title,
            "qas": []  # Empty - Q&A not tied to specific chapters
        }
        
        # Add example to the list
        examples.append(json.dumps(example, ensure_ascii=False) + "\n")
        total_sentences += len(sentences)
        
        print(f"  ✓ {txt_file}: '{title}' ({len(sentences)} sentences)")
        # Print first sentence as sample to verify encoding
        if sentences:
            sample = sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
            print(f"    Sample: {sample}")
    
    # Create Q&A entry (all questions in one entry)
    print("\nProcessing Q&A data...")
    qa_entries = []
    for qa_item in qa_data:
        qa_entry = {
            "question": qa_item.get("question", ""),
            "context": "",  # Vietnamese legal Q&A is document-level
            "answers": [qa_item.get("answer", "")]
        }
        # Add instruction field if present
        if "instruction" in qa_item:
            qa_entry["instruction"] = qa_item["instruction"]
        qa_entries.append(qa_entry)
    
    # Create Q&A example
    qa_example = {
        "file": "qas_full_document.json",
        "sentences": [],  # Empty - no sentences in Q&A entry
        "labels": [],
        "title": "Câu hỏi trắc nghiệm - Luật hành chính công",
        "qas": qa_entries
    }
    
    examples.append(json.dumps(qa_example, ensure_ascii=False) + "\n")
    print(f"  ✓ Processed {len(qa_entries)} Q&A pairs")
    
    # Write examples to the JSONL file
    print(f"\nSaving {len(examples)} examples to {out_file}")
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(examples)
    except Exception as e:
        print(f"❌ Error writing to {out_file}: {e}")
        return None
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {out_file}")
    print(f"{'='*60}")
    print(f"  - Total entries: {len(examples)}")
    print(f"  - Chapter entries: {len(examples) - 1}")
    print(f"  - Total sentences: {total_sentences}")
    print(f"  - Q&A entry: 1 (with {len(qa_entries)} questions)")
    print(f"{'='*60}")
    
    return out_file