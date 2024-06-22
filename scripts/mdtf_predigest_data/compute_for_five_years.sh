#!/bin/bash
for i in {2..9}
do
   python predigest_plots.py 2023 $i
   mv output_2023_0$i.nc /mnt/winds/data/csyhuang/predigest/
done

