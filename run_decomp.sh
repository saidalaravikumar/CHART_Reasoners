#!/bin/bash
for t in street sudoku maze; do python3 decomp.py $t 300 > log_decomp_$t.txt 2>&1; done
