#!/bin/bash

git clone https://github.com/volcengine/verl && cd verl && pip3 install -e .
cd ..
git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
cp verl/patches/megatron_v4.patch  Megatron-LM/
cd Megatron-LM/
git apply megatron_v4.patch
vi megatron_v4.patch
pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $PYTHONPATH
cd ..
cd verl/
vi examples/data_preprocess/gsm8k.py
python
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
cd ..
python download_model.py
pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"
#sh verl/examples/ppo_trainer/run_qwen2.5-0.5b.sh

