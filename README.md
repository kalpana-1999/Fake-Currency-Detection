<b>Introduction:</b><br>
Created a web based Fake currency detection system using machine learning technique and image processing.<br>
<b>Technologies used:</b><br>
Python,OpenCV,NumPy, Flask, Tensorflow<br> 
<b>How it works:</b><br>
step 1- User upload a currency image via web interface<br>
step 2- OpenCV process the image and resize<br>
step 3- trained CNN model analyze feature of the image<br>
step 4- model predict that it is fake or real.<br>
Dataset<br>
- Real 500 rs notes: 110 images (self collected)<br>
- Fake 500 rs notes: 112 images<br>
- Total: 222 images<br>
- Data Augmentation applied to improve model performance<br>
<b>Limitation:</b></br>
- Dataset is small (222 images)n<br>
- Currently works only for 500 rs noten<br>
- Real world performance may vary with image qualityn<br>

