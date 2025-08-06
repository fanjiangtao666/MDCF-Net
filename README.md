# MDCF-Net
This repository contains the implementation of 'MDCF-Net: Modality Decomposition and Compensation Fusion Network for Infrared-Visible Object Detection', a deep learning model designed for infrared and visible light object detection. This work has been accepted by ECAI 2025. If you think this paper can help you, please cite this paper, Thanks!

<img width="4464" height="1623" alt="image" src="https://github.com/user-attachments/assets/4a68b408-f82b-41d9-ae64-ba7c6e92d50c" />


✨✨✨
File Structure and Core Files

## Model

yolov5l_IV_V2.yaml: The model architecture blueprint. This is a YAML configuration file that defines each layer of the network, including the dual-stream backbone, the fusion module, and the detection head.

common_IV_V2.py: Core module definitions. This file defines all the basic PyTorch modules that make up the network, including standard modules borrowed from YOLOv5 and our innovative FusionModule.

yolo_IV_V2.py: The model building script. This script is responsible for parsing the yolov5l_IV_V2.yaml file and dynamically creating the entire network using modules defined in common_IV_V2.py. It defines the Model class, which handles the forward propagation logic for the dual-stream inputs.

## Train

train_IV_V2.py: The main training script. This is the project's entry point, which handles the entire training process, including loading the model, preparing the dataset, and configuring the optimizer and loss function. It specifically integrates the handling of dual-stream inputs and the auxiliary losses returned by the FusionModule.


"""
This file defines the Model class and network parsing logic. It is not intended to be run directly as a script. It should be imported and used by a main training or inference script.
"""

## Dataset
[LLVIP](https://github.com/bupt-ai-cz/LLVIP)
[FLIR](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view)

## Ackowledgement
This code is based on the implementations of [This](https://github.com/bupt-ai-cz/LLVIP) 
