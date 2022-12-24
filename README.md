# Image_Similarity_VGG16
The **VGG16** convolutional neural network (CNN), which is considered to be state of the art for image recognition tasks. We are going to be using this model as a feature extractor only, meaning that we will remove the final (prediction) layer so that we can obtain a feature vector.

•	load_img allows us to load an image from a file as a PIL object
•	img_to_array allows us to convert the PIL object into a NumPy array
•	preproccess_input is meant to prepare your image into the format the model requires. You should load images with the Keras load_img function so that you guarantee the images you load are compatible with the preprocess_input function.
•	VGG16 is the pre-trained model we’re going to use
•	KMeans the clustering algorithm we’re going to use
•	PCA for reducing the dimensions of our feature vector


# Steps Followed
1. Using Pretrained Model **VGG16**, extract only Feature vector(4096 features) by removing final layer.
2. Save result as dict, Key- File_name and value - Feature Vector
3. To Form Clustering - **KNN** used, as we know train computation is high, Moreover we have 4096 features.
4. Inorder to improve training speed, we going to use **PCA(Dimensionality Reduction)**. Reducing feature from 4096 to 100.

# Input - 
  Image folder location , PCA n component and Clustering k value
# Output - 
  Create Folderbased on K value and each folder contain similar images.
  
 References -
 PCA - *https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186#:~:text=Principal%20component%20analysis%2C%20or%20PCA,more%20easily%20visualized%20and%20analyzed.*
 
 VGG16 - *https://www.quora.com/What-is-the-VGG-neural-network*
 
