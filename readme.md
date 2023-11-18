# Cyclone Intensity Estimantion: Using Custom made CNN model
## Tech-Stack: ![Static Badge](https://img.shields.io/badge/Python-ebd31f) ![Static Badge](https://img.shields.io/badge/Tensorflow-f9deb) ![Static Badge](https://img.shields.io/badge/keras-8f1feb) ![Static Badge](https://img.shields.io/badge/Scikit%20Learn-3a1feb) ![Static Badge](https://img.shields.io/badge/Matplotlib-1f9deb) ![Static Badge](https://img.shields.io/badge/Flask-eb1fae) ![Static Badge](https://img.shields.io/badge/HTML5-eb6d1f)

Cyclone intensity estimation over the Indian Ocean is crucial for early preparedness and mitigation of potential damage. This project introduces an automated approach by integrating INSAT satellite imagery with a custom Convolutional Neural Network (CNN) model, enhancing the traditional Dvorak Technique. The model gives an accuracy of 94% with a loss of 20%. 

## Dvorak Technique & Dataset Collection
The Dvorak Technique categorizes cyclones based on T-numbers, reflecting their intensity from mild depression to severe cyclones by analyzing cloud density, eye formation, and tail characteristics. In this initiative, INSAT satellite images were obtained from the MOSDAC (ISRO) site, covering the Indian Ocean region. A dataset comprising images of 15 different cyclones from 2013 to 2021, captured at 30-minute intervals, was meticulously labeled using LabelImg software.

## YOLOv5 Integration 
To enhance automation, the YOLOv5 model was integrated to detect cyclonic clouds in satellite images, facilitating their precise cropping. These preprocessed images are then fed into the custom CNN model for intensity estimation, eliminating the need for human intervention in the process. For more information refer to [YOLO Cyclone Project](https://github.com/nibedita6302/YOLO_Cyclone_Project).

## First time run 

- Create a virtual environment 
```bash 
    python -m venv venv  
```

- Activate the environment 
```bash
    venv\Scripts\activate
```

- Install Flask
```bash
    pip install -r requirements.txt
```

- Run app.py
```bash
    python app.py
```
