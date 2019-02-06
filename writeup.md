## Writeup


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

[original_chessboard]: ./camera_cal/calibration1.jpg
[undistorted_chessboard]: ./output_images/undistorted_calibration1.jpg
[test_image]: ./test_images/test2.jpg
[undistorted_test_image]: ./output_images/undistorted_test2.jpg
[color_and_gradient]: ./output_images/thresholded_binary_test2.jpg
[perspective_transformation]: ./output_images/perspective_transformed_test2.jpg
[poly_fit1]: ./output_images/sliding_window_test2.jpg
[poly_fit2]: ./output_images/sliding_window2_test2.jpg
[result_with_text]: ./output_images/output_with_text_test2.jpg
[video1]: ./test_videos_output/project_video.mp4
[r_curve_formula]: ./examples/r_curve_formula.png
[challenge_example]: /examples/challenge_example.png

[Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in the `Camera Calibration` section of "P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original chessboard image: ![alt text][original_chessboard]

Undistorted chessboard image: ![alt text][undistorted_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This is one of the test images:

![alt_text][test_image]


I applied distortion correction to the test image. I applied `cv2.undistort()` with the camera matrix and the distortion coefficients obtained from the `cv2.calibrateCamera()` function. This gives me the undistorted test image which looks like this:
![alt text][undistorted_test_image]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is located in the `Color and Gradients` section of "P2.ipynb".

I used a combination of color and gradient thresholds to generate a binary image. 
For color transformation, I applied a threshold to the S(Saturation) space because the S channel picks up the lines well, and applied a threshold to the R(Red) space. 
For gradients, I used `cv2.Sobel` and then got the absolute value to accentuate lines away from horizontal.

The I combined the three results as if S channel or gradient is activated, and meanwhile R channel is activated, activate the pixel.

Here's an example of my output for this step.

![alt text][color_and_gradient]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is located in the "Perspective Transformation" section of "P2.ipynb".

The code for my perspective transform includes a function called `warper()`, which appears in the second chunk of this section. The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I computed the perspective transformation matrix using `cv2.getPerspectiveTransform` function and pass it into the `cv2.warpPerspective` function to get the warped binary image.

I verified that my perspective transform was working as expected by seeing that the lines appear parallel in the warped image.

![alt text][perspective_transformation]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial like this:
- Take a histogram of the bottom half of the image.
- Then find the peak of the left and right halves of the histogram, which will be the starting point for the left and right lines.
- Define some parameters like `nwindows`, `margin`, `minpix` which is the minimum number of activated pixels found to move the rectangle (window), and `window_height`.
- When iterate through `nwindows`, draw the boundries of the window, and check if number of activated pixels is greater than `minpix` to deter if we need to move the window.
- Then fit a second order polynomial to all the relevant pixels found in the sliding windows using `np.polyfit`.

The visualization of sliding windows and fitted polynomial drew on the binary image look like this:
![alt text][poly_fit1]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is located in the "Curvature and Position" section of "P2.ipynb".

First I calculated the radius of curvature of the left and right curves using the formula ![alt text][r_curve_formula]

Then I converted them to meters in the real world from pixels using `ym_per_pix = 30/720` and `xm_per_pix = 3.7/700`. 

I took the average of the left and right curvature as the output curvature if they both exist. If only one of the curvature exists and the other one is None, use the existing one.

For calculating the distance to center, I first calculated the x-coordinate of the bottom of the left and right curves by evaluating the fitted line on the y-coordinate. Then find out the center x-coordinate. Then find the distance and convert to meters using the same scale as before.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is located in the "Plot Detected Boundaries" section of "P2.ipynb".

Here I took the fitted polynomial of the left and right curves, and fill in the area in between these two curves on a new image. Then I warped it back to the original image space using `cv2.warpPerspective` with the inverse transformation matrix `M_inverse` obtained from `cv2.getPerspectiveTransform(dst, src)`. 

Then I plotted the warped iamge back to the original image, and added the text to display the curvature and distance from center.

Here is an example of my result on a test image:

![alt text][result_with_text]

---

### Pipeline (video)

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- I didn't use the sliding window with search from prior in the video pipeline, so if no lines are detected in one frame, or of detection is very off in one frame, it won't adjust the result based on prior fitted polynomials.
- When there's an dark edge or shadow near the lane like in the image below, my lane area may be not accurate because my Sobel would detect it as an edge. This can be improved by adjusting the thresholds in gradient and color transform, or apply a region of interest to filter the area of lanes (the former is better since the edit is pretty close to the lane and the it's hard to get an accurate region). ![alt text][challenge_example      ]