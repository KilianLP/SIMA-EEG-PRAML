#!/bin/bash
# Script to delete unused patient data from preprocessed CHB-MIT dataset
# Only keeps patients 1-8 (used for training/validation/testing)
# Deletes patients 9+ to save disk space

REMOTE_DATA="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments/train"

echo "========================================="
echo "CHB-MIT Dataset Cleanup Script"
echo "========================================="
echo ""
echo "This script will delete files for patients 9+ (chb09, chb10, chb11, ...)"
echo "Only keeping patients 1-8 which are used for training."
echo ""
echo "Dataset location: $REMOTE_DATA"
echo ""

# First, count what we have
echo "Counting files..."
TOTAL_FILES=$(find $REMOTE_DATA -name "chb*.pkl" 2>/dev/null | wc -l)
KEEP_FILES=$(find $REMOTE_DATA -name "chb0[1-8]*.pkl" 2>/dev/null | wc -l)
DELETE_FILES=$((TOTAL_FILES - KEEP_FILES))

echo "Current dataset:"
echo "  Total files: $TOTAL_FILES"
echo "  Files to keep (patients 1-8): $KEEP_FILES"
echo "  Files to delete (patients 9+): $DELETE_FILES"
echo ""

if [ $DELETE_FILES -eq 0 ]; then
    echo "No files to delete. Dataset already cleaned up!"
    exit 0
fi

# Calculate space savings
echo "Calculating space usage..."
KEEP_SIZE=$(find $REMOTE_DATA -name "chb0[1-8]*.pkl" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
TOTAL_SIZE=$(find $REMOTE_DATA -name "chb*.pkl" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
echo "  Space used by patients 1-8: $KEEP_SIZE"
echo "  Total space used: $TOTAL_SIZE"
echo ""

# Confirm before deleting
read -p "Do you want to delete $DELETE_FILES files from patients 9+? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Deleting files for patients 9+..."

# Delete files that don't match chb01-chb08
# Using -not -name pattern to delete everything except patients 1-8
find $REMOTE_DATA -name "chb*.pkl" \
    -not -name "chb0[1-8]*.pkl" \
    -print0 | xargs -0 rm -f

echo ""
echo "Cleanup complete!"
echo ""

# Verify cleanup
REMAINING=$(find $REMOTE_DATA -name "chb*.pkl" 2>/dev/null | wc -l)
echo "Verification:"
echo "  Remaining files: $REMAINING"
echo "  Expected: $KEEP_FILES"

if [ $REMAINING -eq $KEEP_FILES ]; then
    echo "  ✓ Cleanup successful!"
else
    echo "  ⚠ Warning: File count doesn't match expected. Please verify manually."
fi
