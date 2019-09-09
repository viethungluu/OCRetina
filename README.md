# OCRetina

# Solutions
This repo provide end-to-end CNNs for text detection and recognition based on RetinaNet. Currently, two solutions are proposed:

## Two-stage OCR system

Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])

![Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])](/images/rosetta.PNG)

Two models are trained independently:
### Text detection model 
A RetinaNet model (with ResNet50 as backbone) is trained to detect word-level text (i.e. "love", "friend"), punctuations, and space character. The model is trained using image randomly generated. The font face is fixed for the first 19 epochs. After that, the font face is randomly chosen.

Sample randomly-generated data

![Sample randomly-generated data](/images/data_detection.png)

Test results after 4 epochs

![Test results after 4 epochs](/images/detection_4.png)

Test results after 19 epochs

![Test results after 19 epochs](/images/detection_19.png)

[Google Colab Notebook for training detection model](https://drive.google.com/file/d/0B7R3L0qnFcRjcHVGRE9HZHNZeFZ5T0RheVJPdlVlVlJGS0xn/view?usp=sharing)

### Text recognition model
A lightweight RNN-based model is trained to recognize text at word level using CTC loss. The model is trained using image randomly generated.

Sample randomly-generated data

![Sample randomly-generated data](/images/data_recognition.png)

[Google Colab Notebook for training recognition model](https://colab.research.google.com/drive/1fEPLZh888mu3NWrXaTYCKrCAfYmO-W-N)

### Existing problems
* While the input shape for detection model can be variable, the input shape for recognition must be fixed. This is due to the limitation of tf.dynamic_rnn which expects a fully-defined feature shape during construction. In this solution, the image is scaled to 128 x 64 (width x height) before feeding to recognition model.

## One-stage OCR system
**THIS SOLUTIONS IS UNDER DEVELOPMENT AND NOT FULLY OPERATED**

One-stage architecture for text detection and recognition in images. Image is adapted from Rosetta's paper (Sorry I'm not good at drawing)

![Proposed one-stage architecture for text detection and recognition in images. Image is adapted from Rosetta's paper (Sorry I'm not good at drawing)](/images/onestage.png)


# References

[1] RetinaNet: https://arxiv.org/abs/1708.02002

[2] CTC loss:  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.139.5852

[3] Rosetta: https://research.fb.com/wp-content/uploads/2018/10/Rosetta-Large-scale-system-for-text-detection-and-recognition-in-images.pdf

# Based sources code

This repo was built using several materials as below.

Keras RetinaNet: https://github.com/fizyr/keras-retinanet

Keras OCR: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

