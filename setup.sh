
#CUDA version = 10.2, GCC = 7.5.0

conda create -n xmf python=3.8 -y 
conda activate xmf
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch -y 
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

pip install -r requirements.txt
pip install setuptools==59.5.0

#now compile all the 3rd party modules in /decoder/utils/

cd decoder/utils/furthestPointSampling
python3 setup.py install

# https://github.com/stevenygd/PointFlow/tree/master/metrics --> not used
cd decoder/utils/metrics/pytorch_structural_losses
make

# https://github.com/sshaoshuai/Pointnet2.PyTorch
cd decoder/utils/Pointnet2.PyTorch/pointnet2
python3 setup.py install

# https://github.com/daerduoCarey/PyTorchEMD
cd decoder/utils/PyTorchEMD
python3 setup.py install

# used to generate partialized partial inputs
cd decoder/utils/randPartial
python3 setup.py install
