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
start=TODO
end=TODO
MEM=32
SIMULRUN=10

if [ "$method" == "LMDEmb" ];then
  MEM=64
fi

declare -a domains=('book')

for domain in "${domains[@]}"
do
  echo "sbatch -p cpu20 -a ${start}-${end}%${SIMULRUN} --mem-per-cpu=${MEM}G -o ${logfolder}${method}_${domain}_${pipeline}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $method $collection;"
  sbatch -p cpu20 -a ${start}-${end}%${SIMULRUN} --mem-per-cpu=${MEM}G -o ${logfolder}${method}_${domain}_${pipeline}.log --open-mode=append run_STATISTICALs_single_complete.sh  $domain $pipeline $method $collection;
  sleep 5;
done