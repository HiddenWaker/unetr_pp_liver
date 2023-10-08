# unetr_pp_liver

This model is from UNETR++, implemented for whole liver segmentation. (CT images)

이 모델은 기존 UNETR++에서 whole liver segmentation을 위해 수정된 버전입니다. (CT 이미지)

---

<p align="center" width="100%">
<img src="https://github.com/HiddenWaker/unetr_pp_liver/assets/132364831/92a0e1fe-8ebf-4a9e-bdb5-c63b150d6c87" alt="IA3 icon" style="width: 1073px; height:360px; display: block; margin: auto; border-radius: 50%;"/>
</p>

---
- This model has been modified to extract only the whole liver from the hepatic vessel CT image.

- To view the original model, follow the link: https://github.com/Amshaker/unetr_plus_plus

# Dataset

To quickly check the performance of the UNETR++ model, we used public data produced by Tian et al. https://github.com/GLCUnet/dataset

---
<p align="left" width="50%">
<img src="https://github.com/HiddenWaker/unetr_pp_liver/assets/132364831/4eabfb8f-e2c5-42ec-b1b6-8c78b34b1140" alt="IA3 icon" style="width: 500px; height:400px; display: block; margin: auto; border-radius: 50%;"/>
</p>

---
  
- Dataset contains annotation of the whole liver.

- We randomly extract 50 nii.gz file from the existing dataset.

- We changed the format of the data to fit the structure of the model. Code for format changing and split (to train and validation) is in [`see_many_files.ipynb`]



# Installation
After cloning the repository, install Anoconda3, follow the below steps for installation,
(If all are pre-installed, you can skip this step.)

- Download Anaconda

- Install PyTorch and torchvision 
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

- Install other dependencies
```
pip install -r requirements.txt
```

# Training

For training, you should run through Anaconda, follow the below steps for installation.

- First, create folder 'unetr_plus_plus-main' on your PC.

- run Anaconda and create virtual enviorment. You can name the virtual environment as you want. 

```
conda activate (put your name of virtual environment)
cd (put your address of folder)\unetr_plus_plus-main
cd (put your address of folder)\unetr_plus_plus-main\training_scripts
sh run_training_synapse.sh
```
- The results of the training are as follows

Batch size | Train loss | Validation loss | Dice score
-- | -- | -- | --
2 | 0.2832 | 0.3127 | 0.9488
**8 (+preprocessing)** | **0.0639** | **0.0896** | **0.9613**
