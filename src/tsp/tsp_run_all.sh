#!/bin/bash
# Script used to run all the traveling salesmen problems with all combinations of parameters.

SAVE=${1:-"False"}
TURNOFF=${2:-"False"}
LOCALLY_LIST="False True"
TOLERANCE_LIST="0.01"
MAX_ITER_LIST="20000 50000"

for MAX_ITER in $MAX_ITER_LIST
  do
    for TOLERANCE in $TOLERANCE_LIST
      do
        for LOCALLY in $LOCALLY_LIST
          do
            sh tsp_run_multiple.sh $MAX_ITER $LOCALLY $SAVE "$TOLERANCE" $MAX_ITER
          done
      done
  done

# Shutdown if finished at night.
if [ "$TURNOFF" = "True" ];
then
  if ! ((8<=$(date +%-H) && $(date +%-H)<23));
  then
      echo shutdown
  fi
fi