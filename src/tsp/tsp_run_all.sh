#!/bin/bash
# Script used to run all the traveling salesmen problems with all combinations of parameters.

LOCALLY_LIST="False True"
TOLERANCE_LIST="0.01"
MAX_ITER_LIST="2000 5000"

for MAX_ITER in $MAX_ITER_LIST
  do
    for TOLERANCE in $TOLERANCE_LIST
      do
        for LOCALLY in $LOCALLY_LIST
          do
            sh tsp_run_multiple.sh $LOCALLY "$TOLERANCE" $MAX_ITER $MAX_ITER
          done
      done
  done