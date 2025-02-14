# .PHONY: install
# See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for issues with torch-scatter, torch-sparse, and torch-cluster
install:
	pip install -e . ;
	pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html ;
	pip install torch_sparse -f  https://data.pyg.org/whl/torch-2.4.0+cu121.html ;
	pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 

