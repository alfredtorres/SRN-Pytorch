# SRN-Pytorch
A Pytorch Implementation of **Structural Relational Reasoning of Point Clouds**. The official reprositity is [here](https://github.com/duanyq14/SRN).  
This project is heavily relied on [Erik's work](https://github.com/erikwijmans/Pointnet2_PyTorch). You should follow Erik's instruction to build the customed opeartions using the following command. 
```
python -m python setup.py build_ext --inplace
```
I change the train.py script to print the loss and acc every step.
## Structural relation module
The SRN module is in `pointnet2_module.py`  
The usage can refer to the pointnet2_ssg_cls.py
```
self.SA_modules.append(
  SRNModule(
    npoints=32,
    gu=[6,6],
    gv=[128,256],
    h=[256+6,128],
    f=[128,128],
    )
  )
```
## Results  
ModelNet40 acc: 0.915316, I use the 12 vote to achieve this. Withour voting, the test acc=0.895462.
## reference
> Yueqi Duan, Yu Zheng, Jiwen Lu, Jie Zhou, and Qi Tian, Structural Relational Reasoning of Point Clouds, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
