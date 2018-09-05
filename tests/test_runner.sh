#!/bin/bash

echo "Testing runner end-to-end"

tmp_embedding_path=$(mktemp -u)
tmp_hypergraph_path=$(mktemp -u)
tmp_metrics_path=$(mktemp -u)

./runner.py \
	--log-level NONE \
	--raw-data test_data/snap_youtube_tiny.cmty.txt \
	--raw-data-format SNAP \
	--embedding $tmp_embedding_path \
	--embedding-method SVD \
	--embedding-dimension 2 \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "End-to-end success!"
else
	echo "End-to-end test failed"
	exit 1
fi

rm -f $tmp_embedding_path

echo "Testing runner premade hypergraph saves embedding"

./runner.py \
	--log-level NONE \
	--embedding $tmp_embedding_path \
	--embedding-method RANDOM \
	--embedding-dimension 2 \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "Premade hypergraph test success!"
else
	echo "Premade hypergraph test failed"
	exit 1
fi

rm -f $tmp_embedding_path
rm -f $tmp_metrics_path
echo "Testing link prediction experiment with leftover hg"

./runner.py \
	--log-level NONE \
	--embedding-method RANDOM \
	--embedding-dimension 2 \
	--experiment-type LINK_PREDICTION \
	--experiment-result $tmp_metrics_path \
	--experiment-lp-probability 0.2 \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "Experiments on loaded hypergraph / embedding test success!"
else
	echo "Experiments on loaded hypergraph / embedding test failed"
	exit 1
fi

rm -f $tmp_embedding_path
echo "Testing new embedding model"

./runner.py \
	--log-level NONE \
	--embedding-method HYPERGRAPH \
	--embedding-dimension 2 \
	--embedding $tmp_embedding_path \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "Keras method success!"
else
	echo "Keras method fail!"
	exit 1
fi

rm -f $tmp_hypergraph_path
rm -f $tmp_embedding_path
rm -f $tmp_metrics_path
