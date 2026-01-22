#!/bin/bash

# --- Configuration ---
WORK_DIR="workdir"

# Check if a file argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 document.txt"
    exit 1
fi

# Store the input file
input_file="$1"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found in $(pwd)"
    exit 1
fi

# --- Create the destination directory ---
if [ ! -d "$WORK_DIR" ]; then
    echo "Creating directory: $WORK_DIR"
    mkdir -p "$WORK_DIR"
fi

# Get the filename without extension and the extension
filename="${input_file%.*}"
extension="${input_file##*.}"

# If the file has no extension, set extension to empty
if [ "$filename" = "$extension" ]; then
    extension=""
    pattern="final"
else
    extension=".$extension"
    pattern="final"
fi

# Define replacement strings for "final"
replacements=("lora" "vera" "uiortholora" "randlora")

# Copy the file 4 times with different replacements
echo "Copying files to $WORK_DIR/..."
for replacement in "${replacements[@]}"; do
    # 1. Replace "final" with the new string in the filename
    new_filename="${filename/${pattern}/${replacement}}${extension}"
    
    # 2. Define the full destination path
    output_file="$WORK_DIR/$new_filename"
    
    # 3. Copy from the current directory to the workdir
    cp "$input_file" "$output_file"
    echo "Created: $output_file"
done

echo "Successfully copied '$input_file' 4 times with replacements into '$WORK_DIR/'"
