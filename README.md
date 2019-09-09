# OCRetina

# Solutions
This repo provide end-to-end CNNs for text detection and recognition based on RetinaNet. Currently, two solutions are proposed:

## Two-stage OCR system

Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])

![Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])](/images/rosetta.PNG)

Two models are trained independently:
* A detection model (RetinaNet) is trained to detect word-level text (i.e. "love", "friend"), punctuations, and space character ([Google Colab Notebook for training detection model](https://drive.google.com/file/d/0B7R3L0qnFcRjcHVGRE9HZHNZeFZ5T0RheVJPdlVlVlJGS0xn/view?usp=sharing))

Test results after 4 epochs
![Test results after 4 epochs](/images/detection.png)

* A RNN based model is trained to recognize word-level text

Note that both models are trained using text image generated from a same vocabulary file.

## One-stage OCR system
**THIS SOLUTIONS IS UNDER DEVELOPMENT. PLEASE TAKE CARE OF YOURSELF USING THIS CODE**

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

