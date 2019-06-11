#download pretrained model
sudo wget http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/resnet/18-layers/resnet-18-0000.params -O model/resnet-18-0000.params
sudo wget http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/resnet/18-layers/resnet-18-symbol.json -O model/resnet-18-symbol.json
sudo wget http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/resnet/50-layers/resnet-50-symbol.json -O model/resnet-50-symbol.json
sudo wget http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/resnet/50-layers/resnet-50-0000.params -O model/resnet-50-0000.params

#dwonload cub .rec data
#sudo wget https://raw.githubusercontent.com/pavanjadhaw/gdown.pl/master/gdown.pl && sudo chmod u+x gdown.pl
sudo ./gdown.pl "https://drive.google.com/file/d/1kgXhtIHx_K57iBVGtwFtMWci1y4KZCeS" data/cub.tar
sudo tar -xvf data/cub.tar -C data/
