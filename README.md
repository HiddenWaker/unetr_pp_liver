# unetr_pp_liver

This model is from UNETR++, implemented for whole liver segmentation. (CT images)

이 모델은 기존 UNETR++에서 whole liver segmentation을 위해 수정된 버전입니다. (CT 이미지)

---

<p align="center" width="100%">
<img src="https://github.com/HiddenWaker/unetr_pp_liver/assets/132364831/92a0e1fe-8ebf-4a9e-bdb5-c63b150d6c87" alt="IA3 icon" style="width: 1073px; height:477px; display: block; margin: auto; border-radius: 50%;"/>
</p>

---
- This model has been modified to extract only the whole liver from the hepatic vessel CT image.

- To view the original model, follow the link: https://github.com/Amshaker/unetr_plus_plus

# Dataset

- To quickly check the performance of the UNETR++ model, we used public data produced by Tian et al. https://github.com/GLCUnet/dataset
  
- Dataset contains annotation of the whole liver.

- We changed the format of the data to fit the structure of the model. Code for format changing and split (to train and validation) is in [`see_many_files.ipynb`] 

# Training

# Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

- Create and activate conda environment
```
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```

- Install PyTorch and torchvision
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

- Install PyTorch and torchvision
```
pip install -r requirements.txt
```

