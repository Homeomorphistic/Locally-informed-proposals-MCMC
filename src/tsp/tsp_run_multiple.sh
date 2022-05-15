#!/bin/bash
# Script used to run multiple traveling salesman problem solutions.

PROBLEMS_LIST="berlin52 kroA150 att532 dsj1000"
LOCALLY=${1:-"False"}
TOLERANCE=${2:-"0.01"}
MAX_ITER=${3:-"1000"}
STAY_LIMIT=${4:-"100"}

for PROBLEM in $PROBLEMS_LIST
  do
    echo "=================================================="
    echo "============== RUNNING $PROBLEM =================="
    echo "============== MAX_ITER=$MAX_ITER ================"
    python3 tsp_solver.py --data "$PROBLEM" --locally "$LOCALLY" --tolerance "$TOLERANCE" --max_iter "$MAX_ITER" --stay_count "$STAY_LIMIT"
  done
