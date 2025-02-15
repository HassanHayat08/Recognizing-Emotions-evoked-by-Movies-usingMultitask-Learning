  # Recognizing Emotions Evoked by Movies Using Multitask Learning  
*(Licensed under the MIT License)*

For details on the associated research, please refer to the paper:  
**Hassan Hayat, Carles Ventura, Agata Lapedriza, “Recognizing Emotions Evoked by Movies Using Multitask Learning”**  
*International Conference on Affective Computing & Intelligent Interaction (ACII 2021)*

This package was developed by Mr. Hassan Hayat (hhassan0@uoc.edu). For any inquiries regarding this package, please feel free to contact him. The package is provided free for academic use and is distributed at your own risk.

---

## Operating System

- **Ubuntu Linux**

---

## Requirements

- **Python:** 3.x.x  
- **GPU:** With CUDA support  
- **Visual Features:** [I3D Model](https://github.com/deepmind/kinetics-i3d)  
- **TensorFlow:** 1.14

---

## Dataset

- **COGNIMUSE Dataset:** [Access Here](http://cognimuse.cs.ntua.gr/database)

---

## Movie Clips Creation and Visual Feature Extraction

### Movie Clips Creation

The COGNIMUSE dataset provides subtitle information along with timestamps, which indicate the starting and ending frames during which each subtitle appears. These timestamps are used to create corresponding movie clips.

### Visual Feature Extraction

An I3D model is utilized to extract visual features from the movie clips. Specifically, the output from the **'Mixed-5c'** layer serves as the visual representation of each frame. Prior to input into the model, each frame is resized to **224x224** pixels.

---

## Setup Instructions

### Single-Task Learning

#### Single-Modality

- **Text Modality:**  
  To train, validate, and test using only the text modality, execute:  
  ```bash
  ./text_modality/st_main.py
  ```

- **Visual Modality:**  
  To train, validate, and test using only the visual modality, execute:  
  ```bash
  ./visual_modality/st_main.py
  ```

#### Multi-Modality

- **Text + Visual Modalities:**  
  To train, validate, and test using both text and visual modalities, execute:  
  ```bash
  ./multi_modality/st_main.py
  ```

### Multi-Task Learning

#### Single-Modality

- **Text Modality:**  
  To train, validate, and test using only the text modality in a multi-task setting, execute:  
  ```bash
  ./text_modality/mt_main.py
  ```

- **Visual Modality:**  
  To train, validate, and test using only the visual modality in a multi-task setting, execute:  
  ```bash
  ./visual_modality/mt_main.py
  ```

#### Multi-Modality

- **Text + Visual Modalities:**  
  To train, validate, and test using both text and visual modalities in a multi-task setting, execute:  
  ```bash
  ./multi_modality/mt_main.py
  ```
