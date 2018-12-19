# Dog Breeds Classifier

## Flowchart
![flowchart](https://i.imgur.com/cKUVat8.png)

## Directory Structure
```
.
├── README.md
├── app.py
├── app_widget.py
├── icon.png
├── model
│   ├── breeds.txt
│   ├── cfg
│   │   ├── darknet53.cfg
│   │   └── yolov3.cfg
│   ├── classification
│   │   ├── [model.h5]
│   │   ├── train.ipynb
│   │   └── train_7_breeds.html
│   ├── data
│   │   ├── crop
│   │   │   └── border_collie
│   │   │       ├── border_collie_000_0.jpg
│   │   │       ├── border_collie_001_0.jpg
│   │   │       └── ...
│   │   └── raw
│   │       └── affenpinscher
│   │           ├── affenpinscher_000.jpg
│   │           ├── affenpinscher_001.jpg
│   │           └── ...
│   ├── font
│   │   ├── FiraMono-Medium.otf
│   │   └── SIL Open Font License.txt
│   ├── model.py
│   ├── model_out.jpg
│   ├── sample
│   │   ├── sample.jpg
│   │   └── sample.mp4
│   ├── yolo
│   │   ├── convert.py
│   │   ├── yolo.py
│   │   ├── yolo_data
│   │   │   ├── coco_classes.txt
│   │   │   ├── [yolo.h5]
│   │   │   └── yolo_anchors.txt
│   │   ├── yolo_model.py
│   │   └── yolo_utils.py
│   └── yolo_detect_and_crop.py
├── requirements.txt
├── search.py
├── sex_classifier
│   ├── haarcascade_frontalface_default.xml
│   ├── make_model.ipynb
│   ├── make_model.py
│   ├── model.h5
│   ├── sex_classification_webcam.ipynb
│   ├── sex_classification_webcam.py
│   └── sex_classification_webcam_qt.py
├── translation.gs
└── view
    ├── gui_test.py
    └── view.py
```

## Model Files
Download from below.
* [model.h5](https://www.dropbox.com/s/q0kyzz67yeixkph/model.h5?dl=0)  
* [yolo.h5](https://www.dropbox.com/s/kozt3gbk5l5ucde/yolo.h5?dl=0)  

## Introduction Slides
* [slides](https://docs.google.com/presentation/d/1l0LN2YL9Yo8Kis8--WkkkermDjyC0KMCYUctEZanW28/edit?usp=sharing)