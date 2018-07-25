# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)




## Outline
---
The goal of this project is to build a pipeline which can conduct vehicle detection in a video. 

The following steps are invovled in developing this pipeline:

* Apply a color transform and append binned color features, as well as histograms of color, and train a classifier Linear SVM classifier 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_new.png
[image2]: ./output_images/hog_new.png
[image3]: ./output_images/window_search_new.png
[image4]: ./output_images/find_car_new.png
[image5]: ./output_images/heatmap_new.png
[video1]: ./project_video.mp4



### Histogram of Oriented Gradients (HOG)
---
The funciton `get_hog_features()` in the cell `Helper Functions` of the jupyter notebook was used to extract HOG features which actually
called the function `skimage.hog()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). In addition, I read [the original HOG paper](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) for the choices of parameters.
For instance, `orient=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` had very decent performance (low miss rates). Considering the image size is (64, 64, 3) for traing a classifier, we decided to use the parameters above.     

Here is an example using HOG parameters of `orient=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


### Linear SVM Classifer
---
The code can be found in the cell `Train a Linear SVM Classifer` of the jupyter notebook was used to extract HOG features which actually
called the function `sklearn.svm.LinearSVC()`. 

The choice of the linear SVM classifer is a balance of performance and cost of running time.  We used 2000 samples of cars and 2000 samples of not-cars to train the classifer to explore the choices of hyperparameters (see above). We also found that color spaces `YUV` and `YCrCb` outperformed other color spaces. The accuracy of the classifer using both`YUV` and `YCrCb` was around `0.99` but the former was a little better. The accuracies we obtain using other color spaces were less than 0.98.  So we decided to use the color space `YUV` here.

As you can see, we normalized the data to zero-mean and unit-variance and splitted the data into the training set (0.8) and validation set (0.2).  We finally used the whole data set (instead of 2000 + 2000 samples) and obtained the accuracy of 0.99 on the validation/test set. 



### Sliding Window Search
---
The functions `slide_window()` and `search_windows()` in the cell `Helper Functions` of the jupyter notebook were used for identifying potential cars. Also. the function `find_cars()` in the cell `Helper Functions` is basically the same idea but considered more efficient (to extract HOG features). You can see how they were used in the code cell `Search Windows`. 

The window search plus classificaiton  work as follows:
* Start from the top-left of the (low-half) image and slide the window (with the size 64x64 and the stride 32 pixels at either x or y direction, i.e., overlap is 0.5). Note that the window size 64x64 should be fixed. But we can try different overlap rates.
* Perdict each window patch we obtain during window sliding to see if it a car or not-car by using the trained linear SVM classifier.
* Save the window patch with positive prediction (i.e., car) and later draw a bounding box for each of them.

Here is an example using `slide_window()` and `search_windows()` with the size 64x64 and overlape rate 0.5:

![alt text][image3]

Note that we actually empolyed `find_cars()` in the pipeline. It used a parameter `scale` to resize the original image (i.e., downsampling if scale > 1). We can play different scales. 

Here is an example using `find_cars()` with `pixels_per_cell=(8, 8)`,  `cells_per_block=(2, 2)` and `cell_per_step=2` (i.e. overlap rate is 0.75):

![alt text][image4]


### Heatmap
---
One way to improve detection is to employ `heatmap`. The functions `add_heat()` and `apply_threshold()` in the cell `Helper Functions` of the jupyter notebook were used to generate `heatmap`.  The basic idea is that a car might be detected by multiple overlapping windows. We can combine these windows to one. Also, it might help remove occasionally misclassified window patches by thresholds.  

Here is an example using `find_cars()` plus `heatmap` and thresholds:

![alt text][image5]

This showed an improvement on previous window search (see above). 


### Pipeline (video)
---
The code for building the pipeline can be found in the cell `Build Vehicle Detection Pipeline` of the jupyter notebook. 
Note that we used two-level filtering to reduce the number of false positives: 
* Using a heatmap and apply a threshold to each frame could filte out some false postives, as shown in the figure above
* Adding `heatmap` over a fixed number of the lastest consecutive frames (e.g., 10 frames) and then applying a threshold to the sum could further filter out additional false positives.

We applied the pipeline to `project_video.mp4` and the output was saved as `processed_project_video.mp4`.


### Discussion
---
We played different scales and heatmap thresholds to see if any improvements could be obtained. But using a list of scales (with not large values) is time-consuming. A samll scale value (e.g., 1.25) plus two-level filtering would help reduce the number of false positives. 

Other ways to improve the performance of the pipeline include:
* building a more robust classifier by, for instance, using more training data.
* exploring other color spaces (or a combination) or other gradient filters (or a combination).

They may deserve further investigation. 
 


