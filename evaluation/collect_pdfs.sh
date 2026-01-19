#!/bin/bash
# Collect all PDF files from gtap/evaluation and organize them in pdf/ directory
# preserving the original directory structure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$SCRIPT_DIR"
PDF_DIR="$EVAL_DIR/pdf"

echo "=== Collecting PDFs from $EVAL_DIR ==="
echo "Destination: $PDF_DIR"
echo ""

# Find all PDF files, excluding the pdf/ directory itself
pdf_files=$(find "$EVAL_DIR" -type f -name "*.pdf" ! -path "$PDF_DIR/*" 2>/dev/null)

if [ -z "$pdf_files" ]; then
    echo "No PDF files found."
    exit 0
fi

count=0
while IFS= read -r src_file; do
    # Get relative path from EVAL_DIR
    rel_path="${src_file#$EVAL_DIR/}"
    
    # Destination path
    dst_file="$PDF_DIR/$rel_path"
    dst_dir="$(dirname "$dst_file")"
    
    # Create destination directory if needed
    mkdir -p "$dst_dir"
    
    # Copy the file
    cp "$src_file" "$dst_file"
    echo "  $rel_path"
    count=$((count + 1))
done <<< "$pdf_files"

echo ""
echo "=== Done: $count PDF files collected ==="
echo ""
echo "Directory structure in $PDF_DIR:"
find "$PDF_DIR" -type f -name "*.pdf" | sed "s|$PDF_DIR/||" | sort

