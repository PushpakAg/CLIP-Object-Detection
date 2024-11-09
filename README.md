# CLIP-Object-Detection

We are doing object detection using input. Here we are using pretrained Faster-RCNN to do region proposal which means it will give us the potential regions where an object can be in an image. Then we use CLIP model to match the selected regions with the text input from user. 

![Result](<test_image.png>)

input = "sitting dog"