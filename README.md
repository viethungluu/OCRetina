# OCRetina

End-to-end one-stage CNNs for text detection and recognition based on RetinaNet. 

**Two-stage OCR system**
Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])

![Facebook's Rosetta system for text detection and recognition in images (source: Facebook \[3\])](/images/rosetta.PNG)

**Proposed one-stage OCR system**
Proposed one-stage architecture for text detection and recognition in images. Image is adapted from Rosetta's paper (Sorry I'm not good at drawing)

![Proposed one-stage architecture for text detection and recognition in images. Image is adapted from Rosetta's paper (Sorry I'm not good at drawing)](/images/onestage.png)


**References**
[1] RetinaNet: https://arxiv.org/abs/1708.02002

[2] CTC loss:  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.139.5852

[3] Rosetta: https://research.fb.com/wp-content/uploads/2018/10/Rosetta-Large-scale-system-for-text-detection-and-recognition-in-images.pdf

**Based sources code**
This repo was built using several materials as below.

Keras RetinaNet: https://github.com/fizyr/keras-retinanet

Keras OCR: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

