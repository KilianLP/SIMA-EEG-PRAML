#!/bin/bash
# Quick check of data size before copying

echo "========================================="
echo "Checking CHBMIT Data Size"
echo "========================================="

DATA_PATH="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments"

echo "Data location: $DATA_PATH"
echo ""
echo "Calculating size (this may take a moment)..."
du -sh $DATA_PATH

echo ""
echo "Number of files:"
find $DATA_PATH -type f -name "*.pkl" | wc -l

echo ""
echo "Breakdown by subdirectory:"
du -sh $DATA_PATH/*

echo ""
echo "Available space in \$SCRATCH:"
if [ -n "$SCRATCH" ]; then
    df -h $SCRATCH | tail -1
else
    echo "\$SCRATCH not set (run this on a compute node with: srun --pty bash)"
fi

echo ""
echo "========================================="
