#!/bin/bash
# Script used to run all the traveling salesmen problems with all combinations of parameters.

SAVE=${1:-"False"}
TURNOFF=${2:-"False"}
PROBLEMS_LIST="berlin52 kroA150 att532 dsj1000"
LOCALLY_LIST="False True"
SCALING_LIST="1 0.1 0.01 0.001"
MAX_ITER_LIST="20000"

echo -e "\n=================================================="
echo -e "============== RUNNING ALL TSP PROBLEMS\n"

START=$(date +%s)
for PROBLEM in $PROBLEMS_LIST
  do
    for SCALING in $SCALING_LIST
      do
        for MAX_ITER in $MAX_ITER_LIST
          do
            for LOCALLY in $LOCALLY_LIST
              do
                echo -e "\n=================================================="
                echo -e "============== RUNNING $PROBLEM"
                echo -e "============== MAX_ITER=$MAX_ITER"
                echo -e "============== SCALING=$SCALING"
                echo -e "============== LOCALLY=$LOCALLY"
                echo -e "============== Started at $(date +%H:%M).\n"
                python3 tsp_solver.py --data "$PROBLEM" --locally "$LOCALLY" --scaling "$SCALING" --max_iter "$MAX_ITER" --stay_count "$MAX_ITER" --save "$SAVE" 2>&1
                echo -e "============== Finished at $(date +%H:%M)."
                echo -e "==================================================\n"
              done
          done
      done
  done
END=$(date +%s)

TIME_ELAPSED=$((END-START))
echo -e "\n============== FINISHED ALL TSP PROBLEMS."
echo -e "============== It took $(((TIME_ELAPSED)/3600)) hours, $(((TIME_ELAPSED)/60)) minutes, $(((TIME_ELAPSED)%60)) seconds."
echo -e "==================================================\n"

# Shutdown if finished at night.
if [ "$TURNOFF" = "True" ];
then
  if ! ((8<=$(date +%-H) && $(date +%-H)<23));
  then
      echo shutdown
  fi
fi