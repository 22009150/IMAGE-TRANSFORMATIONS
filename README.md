# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.

### Step2:

Assign the values to variables in the program.

### Step3:

Get the values from the user appropriately.

### Step4:

Continue the program by implementing the codes of required topics.

### Step5:

Thus the program is executed in google colab.



## Program:
```python
Developed By:Archana k 
Register Number:212222240011
i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt

## Read the input image

input_image = cv2.imread("cat.jpg")

## Convert from BGR to RGB so we can plot using matplotlib

input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

##  Show the image

plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()

ii) Image Scaling

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)

iii)Image shearing
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '3nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)


iv)Image Reflection
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path

image_url = '4 nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)


# Reflect the image vertically

reflected_image_vertical = cv2.flip(image, 0)



# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)


# Display original and reflected images
```
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)

```


v)Image Rotation

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nat5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)




vi)Image Cropping


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '6nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)







```
## Output:


### i)Image Translation
<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/c38054b6-bc50-42cb-9d1b-ca1572a7a3e5)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/c7b2fdac-9dc8-4e88-b74a-f8bef2f8ec30)

<br>
<br>


### iii)Image 
<br>
<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/a4229504-1388-41a8-92cb-39bf4ba20b78)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/a340fc5c-3ec7-4106-b382-58abbffa019f)

<br>


### iv)Image sheamind
<br>
<br>
<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/73f816cf-9ce7-46ca-8b28-1417a7bbfc45)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/1d22c0fc-cdb0-4c0e-ba95-be604c9372b2)




### v)Image reflection
<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/298dc48a-6999-47d9-98dd-c4102cd5d541)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/1886a6ac-3076-4405-b546-ff5e6868c8d2)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/8441b864-364e-4347-8dbb-9024cbb56817)

<br>![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/fcf1a8d3-5a1f-417b-b7c4-3376246878a9)




### v)Image rotation
<br>
<br>
<br>
<br>
### vi) Image cropping 
![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/9466bdd9-0202-4e2e-a8db-77de4ef4c2d7)
![image](https://github.com/22009150/IMAGE-TRANSFORMATIONS/assets/118708624/f71fea87-2b7f-4936-94a0-4d3b3ad77b57)



## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
