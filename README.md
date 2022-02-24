 # This repo an official implement of TGRS2022 paper “PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote Sensing Images”, which is implemented in Pytorch!

 Framework
 -------------------
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/framework.png)

Visualization
---------------------
RSOC:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_RSOC.png)

CARPK:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_CARPK.png)

Crowd:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_Crowd.png)

Result
-----------------------
RSOC:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_RSOC.png)

CARPK:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_CARPK.png)

Drone_crowd:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_Drone.png)
Crowd:
![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_Crowd.png)

##Code
-----------------------
### Install dependencies
-----------------------

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.6

### Train and Test
1. Dowload Dataset
2. Pre-Process Data (resize image and split train/validation)
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>

3、 Train model (validate on single GTX Titan X)
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>
4. Test Model
python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>
5、 Training on ShanghaiTech Dataset
For SHT_A, you should set learning rate to 1e-6, and bg_ratio to 0.1


 
### Pretrain Weight
----------------------

 
 
 
 
 
 
 
 
 
 
The result is slightly influenced by the random seed, but fixing the random seed (have to set cuda_benchmark to False) will make training time extrodinary long, so sometimes you can get a slightly worse result than the reported result, but most of time you can get a better result than the reported one. If you find this code is useful, please give us a star and cite our paper. If having any questions, contact us with the email: gaoguangshuai1990@buua.edu.cn

### Citation
-------------------
@article{gao2022psgcnet,
  title={PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote Sensing Images},
  author={Gao, Guangshuai and Liu, Qingjie and Hu, Zhenghui, and Li, Lu and Wen, Qi and Wang, Yunhong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022}
}


 
 

 
 
 


