#!/usr/bin/env bash

awk '{  if ($7 > 0.1) {split($1,a,"-"); print a[1]"," a[2]"," a[3]"," a[4]"," $3 $5 $7}  }' $1
