#!/bin/bash
source `cat paths_env_vars/virtualenv`
which python

export JAVA_HOME=`cat paths_env_vars/javahomepath`
export PATH="$JAVA_HOME/bin:$PATH"

export CAPREOLUS_LOGGING="DEBUG" ;
export CAPREOLUS_RESULTS=`cat paths_env_vars/capreolusresultpath` ;
export CAPREOLUS_CACHE=`cat paths_env_vars/capreoluscachepath` ;
export PYTHONPATH=`cat paths_env_vars/capreoluspythonpath` ;

# 9 profiles: idxs: 0-8
declare -a profiles=('query' 'basicprofile' 'chatprofile' 'basicprofile_general' 'basicprofile_book' 'basicprofile_book_general' 'chatprofile_general' 'chatprofile_book' 'chatprofile_book_general')

domain=$1
pipeline=$2
method=$3
collection=$4
assessed_set=$5
cf_model=$6
cf_topk=$7
benchmark=kitt


if [ "$method" == "LMD" ];then
  ranker="LMDirichlet"
fi
if [ "$method" == "LMDEmb" ];then
  ranker="LMDirichletWordEmbeddings"
fi
if [ "$method" == "BM25c1.5" ];then
  ranker="BM25 reranker.b=0.75 reranker.k1=1.5 reranker.c=1.5"
fi
if [ "$method" == "BM25cInf" ];then
  ranker="BM25 reranker.b=0.75 reranker.k1=1.5"
fi

# dividant is 10 (folds) * number of queryfilters/cuts which is 1 in this case
dividant=$((10*1))
qtidx=$(( (SLURM_ARRAY_TASK_ID-1)/dividant ))
querytype=${profiles[$qtidx]}
pvidx=$(( SLURM_ARRAY_TASK_ID - (qtidx * dividant)  ))
FOLDNUM=$(( ((pvidx-1)%10)+1 ))

# 9 is number of profiles
if ((SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > (dividant*9))); then
  conda deactivate
  echo "SLURM_ARRAY_TASK_ID is wrong: ${SLURM_ARRAY_TASK_ID}"
  exit
fi

echo "$domain - $pipeline - $querytype - $collection - $method - $cf_model - $cf_topk - $FOLDNUM"

if [ "$pipeline" == "ENTITY_CONCEPT_JOINT_LINKING" ]; then
  if [ "$collection" == "kitt_inferred" ];then
    time ./scripts/capreolus rerank.traineval with benchmark.name=$benchmark benchmark.query_type=$querytype benchmark.assessed_set=$assessed_set benchmark.collection.name=$collection reranker.name=$ranker reranker.extractor.CF_item_lm_top_k=$cf_topk reranker.extractor.CF_model=$cf_model rank.searcher.name=qrels seed=123456 fold=s$FOLDNUM;
  fi
  if [ "$collection" == "kitt" ];then
    time ./scripts/capreolus rerank.traineval with benchmark.name=$benchmark benchmark.query_type=$querytype benchmark.assessed_set=$assessed_set benchmark.collection.name=$collection reranker.name=$ranker rank.searcher.name=qrels seed=123456 fold=s$FOLDNUM;
  fi
fi

conda deactivate
