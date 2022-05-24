#!/bin/bash
# Script used to plot all results of traveling salesmen problem.

PROBLEMS_LIST="berlin52 kroA150 att532 dsj1000"
SAVE=${1:-"False"}

for PROBLEM in $PROBLEMS_LIST
  do
    SCALING_LIST=$(ls "results/sprint-3-test/${PROBLEM}")

    for SCALING in $SCALING_LIST
      do
        SCALING=${SCALING/"scaling="/}
        echo "Plotting ${PROBLEM}, ${SCALING}"
        python tsp_plot.py --data "$PROBLEM" --scaling "$SCALING" --save "$SAVE"
      done
  done