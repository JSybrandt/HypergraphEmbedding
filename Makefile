CC=protoc
MODULE=hypergraph_embedding
PIP=pip3
PYTHON=python3

init: proto pip

proto:
	protoc --python_out=. $(MODULE)/hypergraph.proto

pip:
	$(PIP) install --user -r requirements.txt

test: unittest

unittest:
	$(PYTHON) -W ignore::DeprecationWarning -m unittest discover -v -f

clean:
	rm -f $(MODULE)/hypergraph_pb2.py

format:
	find . -iname "*.py" -exec yapf --style=.style.yapf -i -r {} \;

