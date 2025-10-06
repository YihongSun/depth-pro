#!/bin/bash

for i in {0..7}
do
    python3 run.py -n 8 -i $i &
done
wait