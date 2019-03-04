# Dog Breed Classifier

## Overview
Detect dogs in real time and discriminate breeds.  

## Usage
```
pip3 install -r requirements.txt
python3 app.py [label=path/to/label] [model=path/to/model] [video=path/to/video]
```

## Model File
Download from below.
* [model_8.h5](https://www.dropbox.com/s/ol3w28b8onl23xa/model_8.h5?dl=0)  
* [model_16.h5](https://www.dropbox.com/s/btpeb738uk3mikq/model_16.h5?dl=0)  
* [yolo.h5](https://www.dropbox.com/s/kozt3gbk5l5ucde/yolo.h5?dl=0)  

## Database
* [DynamoDB Local](https://docs.aws.amazon.com/ja_jp/amazondynamodb/latest/developerguide/DynamoDBLocal.DownloadingAndRunning.html)
```
java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -dbPath ./dir/path/to/db
```

## Dataset
* [Dog Breed Identification - Kaggle](https://www.kaggle.com/c/dog-breed-identification)

## Directory Structure
```
.
├── README.md
├── app.py
├── app_widget.py
├── dynamodb
│   ├── data.json
│   ├── dog_breeds.db
│   └── dynamodb.py
├── icon.png
├── model
│   ├── breeds_8.txt
│   ├── breeds_16.txt
│   ├── cfg
│   │   ├── darknet53.cfg
│   │   └── yolov3.cfg
│   ├── classification
│   │   ├── [model_8.h5]
│   │   ├── [model_16.h5]
│   │   ├── train_8.pdf
│   │   ├── train_16.pdf
│   │   └── train.ipynb
│   ├── data
│   │   ├── crop
│   │   │   └── beagle
│   │   │       ├── beagle_000_0.jpg
│   │   │       ├── beagle_001_0.jpg
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
│   ├── sample
│   │   ├── sample.jpg
│   │   └── sample.mp4
│   ├── yolo
│   |   ├── convert.py
│   |   ├── yolo.py
│   |   ├── yolo_data
│   |   │   ├── coco_classes.txt
│   |   │   ├── [yolo.h5]
│   |   │   └── yolo_anchors.txt
│   |   ├── yolo_model.py
│   |   └── yolo_utils.py
|   └── yolo_crop.py  
└── requirements.txt
```

## Screenshot
![screenshot](design/screenshot.png)

## Demo
[![demo](https://img.youtube.com/vi/wk9e7He_Lgc/0.jpg)](https://www.youtube.com/watch?v=wk9e7He_Lgc)
