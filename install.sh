module load stack/2024-06 eth_proxy gcc/12.2.0 cuda/11.8.0
# module load stack/2024-04 eth_proxy gcc/8.5.0 cuda/11.8.0
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'
export TORCH_CUDA_ARCH_LIST="8.6"
export CC=/cluster/scratch/xiychen/miniconda3/bin/gcc
export CXX=/cluster/scratch/xiychen/miniconda3/bin/g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')
# module load stack/2024-05 eth_proxy gcc/13.2.0 cuda/12.2.1
# module load gcc/9.3.0 cuda/11.8.0
# pip install --upgrade --force-reinstall torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu116
# pip install --upgrade --force-reinstall xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu116
pip uninstall -y diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
# pip install git+https://github.com/NVlabs/nvdiffrast
