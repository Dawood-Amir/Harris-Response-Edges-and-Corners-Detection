import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    # Step 1: Compute the gradients Ix and Iy. EXplaination (The gradients IxIx​ and IyIy​ represent the rate of change of intensity at each pixel in the image, in the x and y directions, respectively.) v
    idx= cv2.Sobel(I , cv2.CV_32F , 1,0,ksize=3)
    idy= cv2.Sobel(I , cv2.CV_32F , 0,1,ksize=3)

    # Step 2: Compute the mixed products Ixx, Iyy, Ixy ..EXplaination(Ixx​=Ix2​, Iyy=Iy2Iyy​=Iy2​, and Ixy=Ix×IyIxy​=Ix​×Iy​ represent the auto-correlation of the gradients. These products help us measure how much the image intensity changes in different directions, which is crucial for identifying corners.)
    ixx =idx**2
    iyy = idy**2
    ixy = idx* idy

    # Step 3: Convolve with Gaussian to get A, B, C  . EXplaination(Convolution with a Gaussian filter (using cv2.GaussianBlur) smooths the gradient products. This step reduces noise and helps in stabilizing the detection of corners by averaging out the values in a local neighborhood around each pixel.)
    sigma =1 
    A= cv2.GaussianBlur(ixx,(3,3) , sigmaX=sigma)
    B= cv2.GaussianBlur(iyy,(3,3) , sigmaX=sigma)
    C= cv2.GaussianBlur(ixy,(3,3) , sigmaX=sigma)

    # Step 4: Compute the determinant and trace of the matrix T
    #Tr(M) = a + P = A+B
    #Det(M) = a p = AB - C**@ # from given paper  
    det_T = A*B - C**2
    trace_T = A+B 

    # Step 5: Compute the Harris response R
    #R = Det(T ) − k · Trace(T )2
    R = det_T - k * (trace_T**2)
    
    return R , A, B , C , idx ,idy

 


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """
    To implement the detect_corners function, your goal is to find the stable corner points from the Harris response matrix RR. A corner point is detected if it is both a local maximum in its neighborhood and its value exceeds a certain threshold.
    The approach should be vectorized for efficiency, avoiding loops or external libraries like OpenCV.
    
    Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R =  np.pad(R ,pad_width=1  , mode='constant' , constant_values=0)

    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood
    shifts = [
        (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)
    ]
    neighbours= [np.roll(np.roll(padded_R ,shift[0] ,axis=0), shift[1] , axis=1) for shift in shifts]

    # Step 3 (recommended): Compute the greatest neighbor of every pixel
    max_neighbors =np.maximum.reduce(neighbours)

    # Step 4 (recommended): Compute a boolean image with only all key-points set to True
    # The condition is that the pixel value in R is greater than its neighbors and greater than the threshold
    #R(x, y) > th, with th = 0.1
    keypoint_mask = (R >max_neighbors[1:-1,1:-1]) & (R>threshold) 

    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y_coord, x_coords = np.nonzero(keypoint_mask)

    return x_coords , y_coord


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    padded_R =  np.pad(R , pad_width=1 , mode='constant' , constant_values=0)
    

    # Step 2 (recommended): Calculate significant response pixels
    significant_pixel = (R<=edge_threshold)

    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively
    # Shifted versions for x-axis neighbors
    left_shift  = np.roll(padded_R , 1 , axis=1)
    right_shift =  np.roll(padded_R , -1 , axis=1)

    # Shifted versions for y-axis neighbors
    up_shift =  np.roll(padded_R , 1 ,axis=0)
    down_shift = np.roll(padded_R , -1 , axis=0)

    # Remove padding to align with the original image
    left_shift = left_shift[1:-1, 1:-1]
    right_shift = right_shift[1:-1, 1:-1]
    up_shift = up_shift[1:-1, 1:-1]
    down_shift = down_shift[1:-1, 1:-1]
    

    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors
    x_axis_minima = (R<=left_shift) & (R<=right_shift)
    y_axis_minima = (R <= up_shift) & (R <= down_shift)
    print(x_axis_minima)
    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels
    valid_edge_pixel = significant_pixel &(x_axis_minima | y_axis_minima)

    return valid_edge_pixel
