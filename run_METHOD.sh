#!/bin/bash

logfolder=`cat paths_env_vars/logfolderpath`
pipeline=ENTITY_CONCEPT_JOINT_LINKING

method=$1
if [ "$method" == "" ];then
  echo "give input method LMD LMDEmb BM25c1.5 BM25cInf"
  exit
fi
collection=$2
if [ "$collection" == "" ];then
  echo "give input collection: kitt kitt_inferred"
  exit
fi
cf_model=$3
if [ "$cf_model" == "" ];then
  echo "give input cf_model: description_genre_a100_ns8 description_genre_1-review_a100_ns8 description_genre_all-reviews_a100_ns8"
  exit
fi
cf_topk=$4
if [ "$cf_topk" == "" ];then
  echo "give input cf_topk an int"
  exit
fi
start=$5 # 1
end=$6 # 90
MEM=64
SIMULRUN=10
assessed_set=random20

declare -a domains=('book')

for domain in "${domains[@]}"
do
  echo "sbatch -p cpu20 -a ${start}-${end}%${SIMULRUN} --mem-per-cpu=${MEM}G -o ${logfolder}${method}_${domain}_${pipeline}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $method $collection $assessed_set $cf_model $cf_topk;"
  sbatch -p cpu20 -a ${start}-${end}%${SIMULRUN} --mem-per-cpu=${MEM}G -o ${logfolder}${method}_${domain}_${pipeline}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $method $collection $assessed_set $cf_model $cf_topk;
  sleep 5;
done