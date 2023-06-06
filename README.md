# Face Recognition

- [Description](#description)
- [Architect](#architect)
- [Libraries](#libraries)
- [How to Use the Program](#how-to-use-the-program)

## Description

This is a face recognition program that is trained on [this dataset](https://www.kaggle.com/datasets/saworz/human-faces-with-labels). Please note that the model provided has not been trained, so you will need to train it on your own data.

## Architect

The model architecture is illustrated in the following image:

![architecture](https://github.com/Ali-Fartout/Face-Recognition/blob/main/model.png)

It is a siamese neural network architecture that takes an image as an anchor or input, and another image for comparison, which can be similar (positive) or dissimilar (negative). The model predicts whether the given images are similar or not. For more information, you can refer to the notebooks in the Notebook folder.

## Libraries

The program utilizes the following libraries:

| Libraries | Links                                                                                |
|-----------|--------------------------------------------------------------------------------------|
| PyTorch   | [https://www.pytorch.com](https://www.pytorch.com)                                   |
| NumPy     | [https://numpy.org](https://numpy.org)                                               |
| Pandas    | [https://pandas.pydata.org](https://pandas.pydata.org)                               |
| Pillow    | [https://pillow.readthedocs.io/en/stable/](https://pillow.readthedocs.io/en/stable/) |
| OpenCV    | [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/)   |
| Pathlib   | [https://pypi.org/project/pathlib/](https://pypi.org/project/pathlib/)               |

## How to Use the Program

To use the program, follow these steps:

1. Install Python 3.9 (It is recommended to install this specific version as it was the development environment version), PyTorch, and PIL.

2. Clone the code files by running the following command:

``` 
git clone https://github.com/Ali-Fartout/Face-Recognition.git
```

3. Navigate to the `face_recognition` folder: 
```
cd face_recognition
```

4. Use the following commands for different phases of the program:

- **Training phase**:

  - Create a folder and place all the images of your face into that folder.
  - Copy that folder to the `face_recognition\data` folder.
  - Run the following command to initiate the training phase:
    ```
    python .\app.py "train" "{directory of your image pool}" "{epochs number / default is 5}"
    ```
    For example:
    ```
    python .\app.py "train" "data\musk" "4"
    ```

This will save your model to use multiply times for testing data.

- **Testing phase**:
  - Now import you testing data to this directory : `data\test\input` (Ensure that the number of input images provided for testing  is not less than the size of the image pool.)
  - Use the same image pool folder used for training.
  - Run the following command to perform the testing phase:
    ```
    python .\app.py "test" "{directory of your image pool}" "{owner name}"
    ```
    For example:
    ```
    python .\app.py "test" "data\musk" "musk"
    ```

Note: Do not change any folder names or delete them. If you do so, please retrieve the code from GitHub again.
