#!/bin/bash
# Quick script to see which patients are in the dataset

REMOTE_DATA="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments/train"

echo "Checking patient distribution in CHB-MIT dataset..."
echo "Location: $REMOTE_DATA"
echo ""

# Count files per patient
for i in {01..24}; do
    COUNT=$(find $REMOTE_DATA -name "chb${i}*.pkl" 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        SIZE=$(find $REMOTE_DATA -name "chb${i}*.pkl" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
        USED_IN=$(
            if [ $i -le 5 ]; then echo "(TRAIN)"
            elif [ $i -eq 6 ]; then echo "(VAL)"
            elif [ $i -le 8 ]; then echo "(TEST)"
            else echo "(UNUSED)"
            fi
        )
        printf "Patient %s: %6d files, %8s %s\n" "$i" "$COUNT" "$SIZE" "$USED_IN"
    fi
done

echo ""
echo "Summary:"
USED=$(find $REMOTE_DATA -name "chb0[1-8]*.pkl" 2>/dev/null | wc -l)
UNUSED=$(find $REMOTE_DATA -name "chb*.pkl" -not -name "chb0[1-8]*.pkl" 2>/dev/null | wc -l)
echo "  Files used in training (patients 1-8): $USED"
echo "  Files NOT used (patients 9+): $UNUSED"
