#!/bin/sh

code=./vertical_interpolation_hybrid_to_pressure_poisson_filled_below_topo.py

task(){
   var="$1";
   python $code $var
}

vars=("U" "V" "T")
for element in "${vars[@]}"; do
    task "$element"
done
