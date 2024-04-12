#!/bin/bash


for lambda in {1.8,1.9,2.0,2.1,2.2,0.9,0.8.0.7,0.6}
do

  python3 gan_cond.py $lambda
  echo "NEXT iteation ..."
done
