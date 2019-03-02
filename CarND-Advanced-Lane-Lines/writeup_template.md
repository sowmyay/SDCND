## Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/undist_cb_images/1.jpg "Undistorted"
[image2_1]: test_images/test6.jpg "Original image"
[image2]: output_images/undist_images/test6.jpg "Road Transformed"
[image3]: output_images/thresholded_images/test6.jpg "Binary Example"
[image4]: output_images/warped_images/test6.jpg "Warp Example"
[image4_1]: output_images/unwarped_images/test6.jpg "UnWarp Example"
[image5]: output_images/fit_images/test6.jpg "Fit Visual"
[image6]: output_images/test6.jpg "Output"
[video1]: test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd and 3rd code cells of the IPython notebook located in "P2.ipynb" 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. 

Before applying the undistort function on the image, we refine the camera matrix based on a free scaling parameter using `cv2.getOptimalNewCameraMatrix()`. If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels. So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra black images. It also returns an image ROI which can be used to crop the result.

Notice that without the use of getOptimalNewCameraMatrix function to obtain the camera matrix of the distorted image, a large number of image pixels were being cropped out. This ensures we don't lose out any information from the original image. This function transforms an image to compensate radial and tangential lens distortion

Obtained result: 
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration and distortion coefficients obtained in the previous step,  we apply the distortion correction to the test images using the `cv2.getOptimalNewCameraMatrix()` and  `cv2.undistort()`, as described in previous step.
The code for this step is contained in the 4th code cell of the IPython notebook located in "P2.ipynb" 

Example: We use test1.jpg image provided in test_images folder to demonstrate 
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:


Original image             |  Distortion-corrected image
:-------------------------:|:-------------------------:
![alt text][image2_1]      | ![alt text][image2]

The distortion correction is particularly noticeable in the corners of the image the image, like in the case of the tree near the left edge of the image

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the 5th code cell of the IPython notebook located in "P2.ipynb" , in the `getThresholdImage` function. This code cell also includes a few helper functions that are called with the `getThresholdImage` function.
 
I used a combination of color and gradient thresholds to generate a binary image. I then set a region of interest so that it eliminates other noise in the picture, outside of the lane area.
 
Here's an example of my output for this step.

Original image             |  Thresholded image
:-------------------------:|:-------------------------:
![alt text][image2_1]      | ![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform()`, in the 7th code cell of the IPython notebook. 
The `perspectiveTransform()` function takes as inputs an image (`img`), and the `thresholded` image computed in the previous step.

Since we obtained very good results in lane dection using hough lanes in project 1, we will use this principle in mapping a trapezoid that can be used to warp the image for perspective transform. For this purpose we re-included some of the functions used in project 1 here, in the 6th code cell of the IPython notebook
The `sources` points are thus computed from the lanes dected (`left_line`, `right_line`)
as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


```python
bottom_left = left_line[2:]
bottom_right = right_line[2:]
top_left = left_line[:2]
top_right = right_line[:2]

source = np.float32([bottom_left,bottom_right,top_right,top_left])

h, w, b = img.shape

dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```
I then use `cv2.getPerspectiveTransform(source, dst)` function to obtain the transformation matrix `M`. We then apply the transformation matrix using the `cv2.warpPerspective` function. 
We pass in the image , our transform matrix M , along with the width and height of our output image.

```python
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
image_shape = img.shape
img_size = (image_shape[1], image_shape[0])

warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
unwarped = cv2.warpPerspective(warped, M_inv, img_size , flags=cv2.INTER_LINEAR)
```
I verified that my perspective transform was working as expected by drawing the `source` and `dst` points onto a thresholded test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Thresholded image          |  Warped image             | Unwarped image
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][image3]      | ![alt text][image4]       |  ![alt text][image4_1]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for my finding lane lines includes a function called `search_around_poly()`, in the 8th code cell of the IPython notebook. 
The `search_around_poly()` function takes the binary warped image (`binary_warped`), and the `left_fit` and `right_fit` of the previous iteration, if it exists.

If the we don't have the `left_fit` and `right_fit` from the previous iteration, 
    
    we compute the lanes using the peaks in histogram of the `binary_warped` image and the sliding window concept.
Else

    we compute the lanes using the search from prior method, as described in the lectures
    
    
We then use the `warp_lane()` function, to warp the dected lanes to the original images. You can find this function in the 10th code cell of the IPython notebook.

Lane lines image          | 
:-------------------------:
![alt text][image5]       |


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

We compute the radius of curvature of the lane lines, and the offset of the vehicle using th `measure_curvature_real()` and `distance_from_center` functions respectively. These are included in the 9th code cell in the Ipython notebook.
The radius of curvature in meters was computed using the formula described in the lecture.

In the `distance_from_center()` function, we used the mid point between the fitst left and right fit points
```python
lane_center = (right_fitx[0] + left_fitx[0]) / 2 
center_offset_pixels = abs(img_shape[0] / 2 - lane_center)
center_offset_mtrs = xm_per_pix * center_offset_pixels
``` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

We then use the `warp_lane()` function, to warp the dected lanes to the original images. You can find this function in the 10th code cell of the IPython notebook.
The curvature and offset values computed in the previous step were then added to the warped final image

Final image                |
:-------------------------:|
![alt text][image6]        | 


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest challenge in this project was setting hyperparameters to obtain a thresholded image, that would work well for every scene, irrespective of the road condition and the lighting condtion.
It would have been ideal if we could design a technique that can automatically choose the best suitable hyperparameters based on the scene.

Another scope of improvement is implementing the smoothing function, so that we can eliminate edge cases of incorrect lane detection, using the knowledge of previous snapshots. I will continue to work on implementing this to obtain more refined results.   
