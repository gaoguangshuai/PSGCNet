# PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote Sensing Image
 
 ************************************************************************************************
 
 ###
 The overal framework architecture
 -----------------------------------------------
 ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/framework.png)
 
  ###
  The visualization on RSOC
  -----------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_RSOC.png)
  
  ###
  The visualziation on CARPK
  -----------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_CARPK.png)
  
  ###
  The visualization on crowd counting datasets
  -----------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/visualization_Crowd.png)
  
 
  ###
  The quantitative result on RSOC
  -----------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_RSOC.png)
  
   ###
   The quantitative result on CARPK and PUBCR+
   -----------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_CARPK.png)
  
  ### 
  The quantitative result on DroneCrowd
  -------------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_Drone.png)
  
  ###
  The quantitative result on crowd counting dataset
  --------------------------------------------------
  ![](https://github.com/gaoguangshuai/PSGCNet/blob/main/result_Crowd.png)
  
 
### 
 Code
 ------------------------------------------------------

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.6  

###  Train and Test

1、 Dowload Dataset

2、 Pre-Process Data (resize image and split train/validation)

```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```

3、 Train model (validate on single GTX Titan X)

```
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```

4、 Test Model
```
python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```
The result is slightly influenced by the random seed, but fixing the random seed (have to set cuda_benchmark to False) will make training time extrodinary long, so sometimes you can get a slightly worse result than the reported result, but most of time you can get a better result than the reported one. If you find this code is useful, please give us a star and cite our paper, have fun.

5、 Training on ShanghaiTech Dataset

Change dataloader to crowd_sh.py

For shanghaitech a, you should set learning rate to 1e-6, and bg_ratio to 0.1

**********************************************************************************************************

Paper: https://arxiv.org/abs/2012.03597v3

RSOC Dataset：https://pan.baidu.com/s/19hL7O1sP_u2r9LNRsFSjdA  code：nwcx

or at the website https://drive.google.com/drive/my-drive
but only including building subsets. Other three can be download at https://captain-whu.github.io/DOTA/ according to our provided filenames

CARPK dataset, PUCPR+ dataset: https://lafi.github.io/LPN/

DroneCrowd dataset: https://github.com/VisDrone/DroneCrowd

UCF-QNRF dataset: https://www.crcv.ucf.edu/data/ucf-qnrf/

ShanghaiTech dataset: http://pan.baidu.com/s/1nuAYslz

UCF_CC_50 dataset: https://www.crcv.ucf.edu/data/ucf-cc-50/


***************************************************
References

If you find the PSGCNet useful, please cite our paper. Thank you!

@article{gao2022psgcnet,  
  title={PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote-Sensing Images},  
  author={Gao, Guangshuai and Liu, Qingjie and Hu, Zhenghui and Li, Lu and Wen, Qi and Wang, Yunhong},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  volume={60},  
  pages={1--12},  
  year={2022},  
  publisher={IEEE}  
}










