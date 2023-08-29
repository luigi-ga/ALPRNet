# ALPRNet

ALPRNet is a license plate detection tool based on ASTER. It allows you to efficiently recognize license plates in images.

## Description

ALPRNet is built upon the ASTER OCR (Scene Text Recognition) framework and is specialized for license plate detection. This tool is particularly useful for applications involving vehicle recognition, surveillance systems, and more.

## Installation

To set up ALPRNet, follow these steps:

1. Clone this repository:
```sh
git clone https://github.com/luigi-ga/ALPRNet.git
cd ALPRNet
```
2. Install the required dependencies using `pip`:
```sh
pip install -r requirements.txt
```

## Usage

1. Navigate to the `demo` folder and open the `demo.ipynb` notebook.
2. Inside the notebook, you'll find examples of both training and usage procedures.
3. For training, make sure you have a CSV file with the following header: `filename, label, split`. This CSV file should list the image filenames, corresponding labels, and data split information.

## Pre-trained Model Weights

You can download the pre-trained model weights for both motion-blurred and non-blurred images using the following links:

- [Motion Blurred Weights](https://drive.google.com/file/d/1l2cATgS-tYy46JjUxSxP3n3KwAYLDSSk/view?usp=drive_link)
- [Non-Blurred Weights](https://drive.google.com/file/d/1WZ44A4WIVaMwf2oIzQRXYK-wqsu1vPOo/view?usp=drive_link)

## Acknowledgments

This project is adapted from the [aster.pytorch](https://github.com/ayumiymk/aster.pytorch) repository. The model has been customized for license plate detection. The training process utilized the RodoSol-ALPR and AOLP datasets, along with a custom dataset.


---

Feel free to reach out if you have any questions or suggestions. You can contact me [here](mailto:gallo.1895146@studenti.uniroma1.it).


