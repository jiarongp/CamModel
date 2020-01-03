# CamModel
Research Project of CNN-based Camera Model Identification

## Download Dresden Database
1. see if the data path already exist
2. if not, then make a new one, if yes just download the database
3. download urls are read from the dresden.csv

For this experiment, I only use the brand **Afga** and **Canon**.
- Afga: DC-504(169), DC-733S(281), DC-830I(363), 505-X(172), 530S(372)
- Canon: IXUS 55(224), IXUS 70(567), A640(188)

In totoal, there are 2336 images.

## Extract Patches
each model has at least 150 images
1. randomly select 80% of the database
2. divide these images in to 256 x 256 pixel patches
3. retain the central 25 patches from the green layer of each image for training databse

## Data preparation
Format the images into appropriately pre-processed floating point tensors before feeding to the network:
1. Read images from the disk.
2. Decode contents of these images and convert it into proper grid format as per their RGB content.
3. Convert them into floating point tensors.
4. Rescale the tensors from values between 0 and 255 to values between 0 and 1, as **neural networks prefer to deal with small input values.**