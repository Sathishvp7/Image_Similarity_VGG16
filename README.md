# Image_Similarity_VGG16
The VGG16 convolutional neural network (CNN), which is considered to be state of the art for image recognition tasks. We are going to be using this model as a feature extractor only, meaning that we will remove the final (prediction) layer so that we can obtain a feature vector.

load_img allows us to load an image from a file as a PIL object
img_to_array allows us to convert the PIL object into a NumPy array
preproccess_input is meant to prepare your image into the format the model requires. You should load images with the Keras load_img function so that you guarantee the images you load are compatible with the preprocess_input function.
VGG16 is the pre-trained model we’re going to use
KMeans the clustering algorithm we’re going to use
PCA for reducing the dimensions of our feature vector
