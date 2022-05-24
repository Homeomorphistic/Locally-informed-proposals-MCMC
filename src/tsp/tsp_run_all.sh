#!/bin/bash
# Script used to run all the traveling salesmen problems with all combinations of parameters.

SAVE=${1:-"False"}
TURNOFF=${2:-"False"}
LOCALLY_LIST="False True"
SCALING_LIST="1 0.1 0.01 0.001"
MAX_ITER_LIST="1000 2000 5000"

for MAX_ITER in $MAX_ITER_LIST
  do
    for SCALING in $SCALING_LIST
      do
        for LOCALLY in $LOCALLY_LIST
          do
            sh tsp_run_multiple.sh $MAX_ITER $LOCALLY $SAVE "$SCALING" $MAX_ITER
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