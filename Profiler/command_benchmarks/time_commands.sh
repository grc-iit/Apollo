#!/bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "--------------------------------------------------------------------------------"
    echo $line
    time $line
done < "$1"
