#!/bin/bash
# Script used to plot all results of traveling salesmen problem.

PROBLEMS_LIST="berlin52 kroA150"
SPRINT=${1:-"7-low-iter"}
SAVE=${2:-"False"}

for PROBLEM in $PROBLEMS_LIST
  do
    TEMPERATURE_LIST=$(ls "results/sprint-${SPRINT}/${PROBLEM}/locally=False")

    for TEMPERATURE in $TEMPERATURE_LIST
      do
        COOLING_LIST=$(ls "results/sprint-${SPRINT}/${PROBLEM}/locally=False/${TEMPERATURE}")
        TEMPERATURE=${TEMPERATURE/"temp="/}
        for COOLING in $COOLING_LIST
          do
            COOLING=${COOLING/"cool="/}
            echo "Plotting ${PROBLEM}, temp=${TEMPERATURE}, cool=${COOLING}"
            python tsp_plot.py --sprint "$SPRINT" --data "$PROBLEM" --temperature "$TEMPERATURE" --cooling "$COOLING" --save "$SAVE"
          done
      done
  done