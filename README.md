# Pytorch codes of Category Semantic Guided Unsupervised Domain Adaptation Network for Hyperspectral Image Classification
![image](https://github.com/user-attachments/assets/adb1ada2-6cf6-481b-aca8-3956aab8c562)
# Requirements
CUDA Version: 11.3 <br>
torch: 1.11.0 <br>
Python: 3.8.10 <br>
# Dataset
You can download the hyperspectral datasets in mat format at:, and move the files to ./datasets folder. <br>
An example dataset folder has the following structure: <br>
datasets
├── Houston
│   ├── Houston13.mat
│   └── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_gt_7.mat
│   ├── pavia.mat
│   └── pavia_gt_7.mat
│── Shanghai-Hangzhou
│   ├── Shanghai.mat
│   └── Shanghai_gt.mat
│   ├── Hangzhou.mat
│   └── Hangzhou_gt.mat
