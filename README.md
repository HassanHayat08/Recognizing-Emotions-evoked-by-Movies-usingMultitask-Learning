# Source code of Recognizing Emotions evoked by Movies using Multitask Learning. (The Code Follows MIT License) #
To read the paper: Hassan Hayat, Carles Ventura, Agata Lapedriza, “Recognizing Emotions evoked by Movies using Multitask Learning”, International Conference on Affective Computing & Intelligent Interaction (ACII 2021)

This package was developed by Mr.Hassan Hayat (hhassan0@uoc.edu). Please feel free to contact in case of any query regarding the package. You can run this package at your own risk. This package is free for academic use.

**Operating System** 

- Ubuntu Linux

**Requirements**

- Python 3.x.x

- GPU with CUDA support

- Visual features [I3D Model](https://github.com/deepmind/kinetics-i3d)

- Tensorflow 1.14

**Dataset**

- Dataset [COGNIMUSE](http://cognimuse.cs.ntua.gr/database) 

## Movie clips creation and visual feature extraction information 
**How to create movie clips of the COGNIMUSE dataset?**

The subtitle information of movies is provided with the timestamps. These timestamps represent the starting and ending frame information in which the subtitle appears. We used these timestamps to create clips.

**How to get the visual features?**

We used an I3D model to get the visual features. The output of the ‘Mixed-5c’ layer of the model presents the visual representation of the given frame. Each frame was transformed into 224x224 sizes before feeding into the model.

## Setup
**Single-Task Learning**

**Single-Task (ST) Learning using Single-Modality**

*train, validate, and test the single-task learning using only text modality*

- ./text_modality/st_main.py

*train, validate, and test the single-task learning using only visual modality*

- ./visual_modality/st_main.py

**Single-Task (ST) Learning using Multi-Modality**

*train, validate, and test the single-task learning using text+visual modalities*

- ./multi_modality/st_main.py

**Multi-Task Learning**

**Multi-Task (MT) Learning using Single-Modality**

*train, validate, and test the multi-task learning using only text modality*

- ./text_modality/mt_main.py

*train, validate, and test the multi-task learning using only visual modality*

- ./visual_modality/mt_main.py

**Multi-Task (MT) Learning using Multi-Modality**

*train, validate, and test the multi-task learning using text+visual modalities*

- ./multi_modality/mt_main.py
