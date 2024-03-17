# human-brain-tomer-image-segmentaion
A brain tumor refers to the abnormal growth of cells either within the brain tissue itself or in close proximity to it. These tumors may develop within the brain or adjacent structures such as nerves, the pituitary gland, the pineal gland, and the brain's protective membranes
#c<div align="center">
      <h1> <img src="Arif_Miah952" width="80px"><br/>introduction to </h1>
     </div>
<p align="center"> <a href="Arif_Miah952" target="_blank"><img alt="" src="https://img.shields.io/badge/Twitter-1DA1F2?style=normal&logo=twitter&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.kaggle.com/arifmiad}" target="_blank"><img alt="" src="https://img.shields.io/badge/LinkedIn-0077B5?style=normal&logo=linkedin&logoColor=white" style="vertical-align:center" /></a> </p>

# Description
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. Initially developed by Intel in 1999, it has since become a widely used tool in various fields such as robotics, augmented reality, image and video processing, and more. Offering a comprehensive suite of functions, OpenCV enables developers to perform tasks like object detection, facial recognition, image segmentation, and motion tracking with ease. Its extensive support for multiple programming languages including C++, Python, Java, and more, makes it accessible to a broad community of developers. With its efficient algorithms and constant updates, OpenCV remains a cornerstone in the development of cutting-edge computer vision applications

# Features

In OpenCV, a "feature" generally refers to distinctive attributes or characteristics of an image that can be used for various tasks such as object detection, recognition, tracking, and matching. These features are typically extracted from an image using algorithms like 1.SIFT (Scale-Invariant Feature Transform), 
2.SURF (Speeded-Up Robust Features),
3 ORB (Oriented FAST and Rotated BRIEF).


# Tech Used
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white) ![MySQL](https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
      
# Import important libray :
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt




```


### I can  add to setup:
- Step 1: load image_path
- Step 2: read image
``` python

```

# Find out  orginal image to edge and contours of image
``` python
blur = cv2.GaussianBlur(image, (5,5),0)
edge = cv2.Canny(blur, 100, 200)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
plt.figure(figsize = (10, 12))


plt.subplot(1, 3, 1)
plt.imshow(image, cmap = "gray")
plt.title("Orginal Image")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(edge, cmap = "gray")
plt.title("Edge Image")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(contour_image)
plt.title("Contours")
plt.axis("off")


plt.show()
```
![image](https://github.com/Arif-miad/human-brain-tomer-image-segmentaion/assets/83044522/b257e86e-2ed8-4906-a8b6-5eaa5dffa50c)

```python
image_path = "/content/Image.png"
image = cv2.imread(image_path)
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, binary = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)


gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


fig, axs = plt.subplots(2, 2, figsize = (10, 10))



axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Orginal Image")
axs[0, 0].axis("off")


axs[0, 1].imshow(gray, cmap = "gray")
axs[0, 1].set_title("GraY Image")
axs[0, 1].axis("off")


axs[1, 0].imshow(binary, cmap = "gray")
axs[1, 0].set_title("BINARY Image")
axs[1, 0].axis("off")

axs[1, 1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title("Contour Image")
axs[1, 1].axis("off")


plt.show()
```
![image](https://github.com/Arif-miad/human-brain-tomer-image-segmentaion/assets/83044522/9916b34d-db0e-4ac0-9c5f-565fe691709d)

 
```
i am  using K-means clustering for image edge connection contouns and describe the image and find more imformation this image
```
 


      
<!-- </> with 💛 by readMD (https://readmd.itsvg.in) -->
    
