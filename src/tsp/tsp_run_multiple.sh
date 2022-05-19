#!/bin/bash
# Script used to run multiple traveling salesman problem solutions.

PROBLEMS_LIST="berlin52 kroA150 att532 dsj1000"
MAX_ITER=${1:-"1000"}
LOCALLY=${2:-"False"}
SAVE=${3:-"False"}
TOLERANCE=${4:-"0.01"}
STAY_LIMIT=${5:-$MAX_ITER}


for PROBLEM in $PROBLEMS_LIST
  do
    echo "\n=================================================="
    echo "============== RUNNING $PROBLEM"
    echo "============== MAX_ITER=$MAX_ITER"
    echo "============== Started at $(date +%H:%M).\n"
    python3 tsp_solver.py --data "$PROBLEM" --locally "$LOCALLY" --tolerance "$TOLERANCE" --max_iter "$MAX_ITER" --stay_count "$STAY_LIMIT" --save "$SAVE"
    echo "============== Finished at $(date +%H:%M)."
    echo "==================================================\n"
  done
