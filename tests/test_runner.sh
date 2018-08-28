#!/bin/bash

echo "Testing runner end-to-end"

tmp_embedding_path=$(mktemp -u)
tmp_hypergraph_path=$(mktemp -u)

./runner.py \
	--raw-data test_data/snap_youtube_tiny.cmty.txt \
	--raw-data-format SNAP \
	--embedding $tmp_embedding_path \
	--embedding-method SVD \
	--dimension 2 \
	--log-level NONE \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "End-to-end success!"
else
	echo "End-to-end test failed"
	exit 1
fi

rm -f $tmp_embedding_path

echo "Testing runner premade hypergraph"

./runner.py \
	--embedding $tmp_embedding_path \
	--embedding-method RANDOM \
	--dimension 2 \
	--log-level NONE \
	$tmp_hypergraph_path

if [ $? -eq 0 ]; then
	echo "Premade hypergraph test success!"
else
	echo "Premade hypergraph test failed"
	exit 1
fi

rm -f $tmp_embedding_path
rm -f $tmp_hypergraph_path
