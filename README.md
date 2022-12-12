# Motor Classification and Segmentation
This project is a sub-project of the research project **AgiProbot** from KIT and Bosch. We develop a benchmark including 2D synthetic image datasets and 3D synthetic point cloud datasets. In this part, classification and segmentation are trained jointly, the results of separate training are also provided.

<img src="https://github.com/LinxiQIU/motor_seg_cls/blob/main/images/cls_seg.png" width="800" height="400">

## Environment Requirements

CUDA = 10.2

Python >= 3.7.0

PyTorch = 1.6

The mentioned API are the basic API. In the training process,if there is warning that some modul is missing. you could direct use pip install to install specific modul.

## 3D Classification

The task of classification is to classify the key object in the scene into the prior-defined categories. There are 5 types of motors in our dataset, the main differences between them are the number of gears and the shape of covers.

```python
CUDA_VISIBLE_DEVICES=6,7 python main_cls.py --exp_name classification --change adamw --root /home/ies/qiu/dataset/dataset1000
```

Explanation of the important parameters:

* CUDA_VISIBLE_DEVICES: set the visible gpu

* main_cls.py: choose of which script will be run

* exp_name: the paremeter (classification) means the results of the classification task

* change: give the information of a specific experiment(e.g. the changes of batch_size, epochs, optimizer or learning rate)

* root: the root directory of training dataset

## 3D Segmentation

3D point cloud segmentation is the process of classifying point clouds into different regions, so that the points in the same isolated region have similar properties. Common segmentation metrics are category-wise IoU and overall mIoU. In our case, the overall mIoU metric is used.

### Train

```python
CUDA_VISIBLE_DEVICES=6,7 python main_semseg.py --exp_name semseg --change adamw --root /home/ies/qiu/dataset/dataset1000
```

### Test

After training the model, you can use the following command to get the test result.

```python
CUDA_VISIBLE_DEVICES=6,7 python main_semseg.py --root /home/ies/qiu/dataset/dataset1000 --eval True --model_path /home/ies/qiu/motor_seg_cls/outputs/dgcnn/semseg/adamw/models/best_m.pth
```

## 3D Classification and Segmentation

classification and segmentation are trained jointly.

```python
CUDA_VISIBLE_DEVICES=6,7 python main_cls_seg.py --exp_name semseg --change adamw --root /home/ies/qiu/dataset/dataset1000
```