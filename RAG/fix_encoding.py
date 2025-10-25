#!/usr/bin/env python3
"""
Utility script to fix encoding issues in Vietnamese .txt files
Converts files to proper UTF-8 encoding
"""

import os
import sys
from pathlib import Path


def detect_and_convert_encoding(file_path):
    """
    Detect encoding and convert file to UTF-8
    """
    # Common encodings for Vietnamese text
    encodings_to_try = [
        'utf-8',
        'utf-8-sig',      # UTF-8 with BOM
        'latin-1',        # ISO-8859-1
        'cp1252',         # Windows-1252
        'iso-8859-1',
    ]
    
    content = None
    detected_encoding = None
    
    # Try to read with different encodings
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            detected_encoding = encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        return False, "Could not decode file with any known encoding"
    
    # Check if it's already proper UTF-8 (without BOM)
    if detected_encoding == 'utf-8':
        # Verify it's actually UTF-8 and not misdetected
        try:
            content.encode('utf-8')
            return True, "Already UTF-8"
        except UnicodeEncodeError:
            pass
    
    # Convert to UTF-8
    try:
        # Create backup
        backup_path = str(file_path) + '.backup'
        if not os.path.exists(backup_path):
            with open(file_path, 'rb') as f_in:
                with open(backup_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        # Write as UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, f"Converted from {detected_encoding} to UTF-8"
    except Exception as e:
        return False, f"Error: {e}"


def fix_folder(folder_path):
    """
    Fix all .txt files in a folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder '{folder_path}' does not exist!")
        return
    
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in '{folder_path}'")
        return
    
    print(f"Found {len(txt_files)} .txt files in '{folder_path}'")
    print("=" * 60)
    
    success_count = 0
    failed_count = 0
    
    for txt_file in txt_files:
        success, message = detect_and_convert_encoding(txt_file)
        
        if success:
            print(f"✅ {txt_file.name}: {message}")
            success_count += 1
        else:
            print(f"❌ {txt_file.name}: {message}")
            failed_count += 1
    
    print("=" * 60)
    print(f"Summary: {success_count} succeeded, {failed_count} failed")
    
    if success_count > 0:
        print("\n💡 Backup files created with .backup extension")
        print("   If everything looks good, you can delete them:")
        print(f"   rm {folder_path}/*.backup")


def fix_single_file(file_path):
    """
    Fix a single file
    """
    if not os.path.exists(file_path):
        print(f"❌ File '{file_path}' does not exist!")
        return
    
    print(f"Processing: {file_path}")
    success, message = detect_and_convert_encoding(file_path)
    
    if success:
        print(f"✅ {message}")
        print(f"💡 Backup created: {file_path}.backup")
    else:
        print(f"❌ {message}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Fix all .txt files in a folder:")
        print("    python fix_encoding.py /path/to/folder")
        print()
        print("  Fix a single file:")
        print("    python fix_encoding.py /path/to/file.txt")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        fix_folder(path)
    elif os.path.isfile(path):
        fix_single_file(path)
    else:
        print(f"❌ '{path}' is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()