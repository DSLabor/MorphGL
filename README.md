# MorphGL


Installation:
```
# basic setup
conda create -n morph python=3.9
conda activate morph
pip install torch==2.0.1 numpy pandas packaging ogb numba

# install customized dgl of ducati
git clone https://github.com/initzhang/dc_dgl.git
cd dc_dgl
git checkout mix
sh mybuild.sh

# install pyg required by salient
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# install salient
cd third_party/salient/fast_sampler
python setup.py install
```

Then run experiments with `python example_usage.py`. Check all valid arguments in `parser.py`.
