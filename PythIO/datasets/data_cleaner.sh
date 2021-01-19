#! /usr/bin/env bash

echo "model,metric_dataset,r2,mae,mse,bigInference" > $1_r2_filtered.csv
echo "model,metric_dataset,r2,mae,mse,bigInference" > $1_inference.csv
echo "model,metric_dataset,r2,mae,mse,bigInference" > $1_r2_mse.csv

sed_cmd=$(sed -e 's/^hdd-/hdd_/g' -e 's/Avgrq-sz/Avgrq_sz/g' -e 's/^ssd-/ssd_/g' -e 's/^all-/all_/g'   -e 's/^nvme-/nvme_/g' -e 's/50-50-LSTM.model/50_50_LSTM.model/g' -e 's/-5-[0-9]*/5_model/g' $1)

echo "$sed_cmd" | awk 'NR%2  {  if (length($0) > 20 && $3 != "nan," ) {split($1,a,"-"); print a[1]"," a[2]"_" a[3]"_" a[4]"," $3 $5 $7 $9 $10 $11}  }'  >> $1_r2_mse.csv
echo "$sed_cmd" | awk '      {  if (length($0) > 20 && $3 ~ "nan"   ) {split($1,a,"-"); print a[1]"," a[2]"_" a[3]"_" a[4]"," $3 $5 $7 $9 $10 $11}  }'  >> $1_inference.csv
echo "$sed_cmd" | awk 'NR%2  {  if (length($0) > 20 && $3 > 0       ) {split($1,a,"-"); print a[1]"," a[2]"_" a[3]"_" a[4]"," $3 $5 $7 $9 $10 $11}  }'  >> $1_r2_filtered.csv

