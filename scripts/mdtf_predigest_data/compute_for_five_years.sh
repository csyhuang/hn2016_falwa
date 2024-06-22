#!/bin/bash
for year in {2020..2022}
do
  for i in {1..9}
  do
     python predigest_data.py $year $i
     mv output_$year_0$i.nc /mnt/winds/data/csyhuang/predigest/
  done
  for i in {10..12}
  do
     python predigest_data.py $year $i
     mv output_$year_$i.nc /mnt/winds/data/csyhuang/predigest/
  done
done



  for i in {10..12}
  do
     python predigest_data.py 2015 $i
     mv output_2015_$i.nc /mnt/winds/data/csyhuang/predigest/
  done