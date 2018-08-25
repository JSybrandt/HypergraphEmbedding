CC=protoc
MODULE=hypergraph_embedding

init: proto pip

proto: pip
	protoc --python_out=. $(MODULE)/hypergraph.proto

pip:
	pip3 install --user numpy
	pip3 install --user scipy
	pip3 install --user protobuf

test:
	python3 -m unittest discover -v 

clean:
	rm -f $(MODULE)/hypergraph_pb2.py
