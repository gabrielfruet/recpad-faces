#!/bin/bash
# Usage: ./view_csv_table.sh file.csv

if [ $# -ne 1 ]; then
    echo "Usage: $0 file.csv"
    exit 1
fi

csvfile="$1"
if [ ! -f "$csvfile" ]; then
    echo "File not found: $csvfile"
    exit 2
fi

# Display CSV as a justified table
cat "$csvfile" | column -s, -t
