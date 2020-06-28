sudo apt-get install -y libopenblas-dev liblapack-dev libopencv-dev
sudo ln /dev/null /dev/raw1394


sudo chmod u+x gdown.pl
sudo ./gdown.pl "https://drive.google.com/file/d/1ZWHE6vXD84G-dkB6bhSC4PLar59NHoMF" mxnet_python.tar
sudo tar -xvf mxnet_python.tar 
cd mxnet_python/python
sudo pip install -e .

#down load nccl for cuda 8.0
cd ../../
sudo ./gdown.pl "https://drive.google.com/file/d/11kx-ElMilj_3uxVQeBR_Zgiq2dUyZGsU" nccl.tar
sudo tar -xvf nccl.tar

#down load cudnn for cuda 8.0
sudo ./gdown.pl "https://drive.google.com/file/d/1n2-9du8dCXavK5VfjPtXOHnfJ5fophT1" cudnn-8.0-linux-x64-v5.0-ga.tgz
sudo tar zxvf cudnn-8.0-linux-x64-v5.0-ga.tgz
