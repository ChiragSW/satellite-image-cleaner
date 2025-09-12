# Cloud Cleaning and Enhancement model
This model detects clouds in a satellite image using a made-from-scratch ResNet18 model. Cloud segmenter identifies the cloud area in image and then GatedPartialConv cleans the clouds from images. The image is enhanced using RRDB (ESRGAN).

## Table of Contents
- [Tech Stack](#tech-stack-used)  
- [Model Structure](#model-structure)  

## Tech Stack used
| Stack | Tech |
|----------|----------|
| Language | ![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python&logoColor=white) |
| Data Handling    | ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-221f1f?logo=matplotlib&logoColor=white)|
| ML    | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)  |
| Vision    | ![Torchvision](https://img.shields.io/badge/Torchvision-%23EE4C2C.svg?logo=PyTorch&logoColor=white)  |
| File handling | ![Pathlib](https://img.shields.io/badge/Pathlib-3776AB?logo=python&logoColor=white) ![Shutil](https://img.shields.io/badge/Shutil-3776AB?logo=python&logoColor=white) |
| Datasets | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white) |
| Training | ![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&logoColor=white) ![VSCode](https://img.shields.io/badge/VSCode-0078D7?logo=visual-studio-code&logoColor=white) |
| Pretrained weights | ![ResNet18](https://img.shields.io/badge/ResNet18-CloudClassifier-blueviolet) ![UNet](https://img.shields.io/badge/UNet-CloudSegmenter-lightblue) ![RRDBNet](https://img.shields.io/badge/RRDBNet-Enhancement-green) |

## Model structure
```
satellite-image-cleaning/
│
├── checkpoints/                    # best weights after training 
│   ├── cloud_classifier_best.pth   # generated after running train_classifier.py
│   ├── cloud_segmenter.pth         # generated after running train_segmenter.py 
│   ├── gated_conv_inpainting.pth   # generated after running train_inpainter.py
│   └── RRDB_ESRGAN_x4.pth          # pretrained RRDB weights (download from github)
│   
├── data/                           # contains cloudy and clear images data     
│   ├── train/ 
|       ├── clear/
|       └── cloudy/
│   ├── val/
|       ├── clear/
|       └── cloudy/
|     
├── inpainting_data/                # contains images and their masks to learn inpainting    
│   ├── images/
|   └── masks/
|
├── models/
|   ├── cloud_classifier.py         # classifies if image has clouds
|   ├── cloud_segmenter.py          # identifies the clouds in image
|   ├── esrgan.py                   # model for enhancement
|   ├── partialconv_network.py      # gated partial conv network
|   └── partialconv.py              # gated partial conv model
|
├── partial_conv                    # partial conv repo from github
|                                      
├── satellite_env                   # virtual env
|           
├── segmentation_data/              # contains images and their masks to learn cloud segmentation    
│   ├── images/
|   └── masks/
|
├── testing-env/                    # contains input and output images
|   ├── inputs/
|   └── outputs/
|
├── utils/                          # preprocessing
|   ├── prepare_dataset.py
|   └── preprocess.py
|
├── .gitignore    
├── LICENSE
├── README.md
|
├── pipeline.py                     # model pipeline
├── pipeline512.py                  # robust model that outputs high res image (uses more than 15GB memory GPU)
├── requirements.txt
├── run_pipeline.py                 # used to run the model
├── train_classifier.py             # train classifier model
├── train_inpainter.py              # train inpainter model
└── train_segmenter.py              # train segmenter model
```
