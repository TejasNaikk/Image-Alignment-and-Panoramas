# Image-Alignment-and-Panoramas
Stitching different perspective images into a single smooth panorama using Laplacian Blending. 
Homographic and Affine Transformation were used to create Perspective and Cylindrical warping respectively. The SIFT feature descriptors of the images are then matched together and blended to form a single panoramic view.

Uploaded files Description:

main.py: Python code implementing Perspective and Cylindrical warping to create a smooth panorama.

input1.png: Input image 1 for creation of panorama.

input2.png: Input image 2 for creation of panorama.

input3.png: Input image 3 for creation of panorama.

output-cylindrical.png: Result of perspective warping (Homographic Transormation)

output-cylindrical-lpg.png: Result of perspective warping (Homographic Transormation) with Laplacian Blending

output-homography.png: Result of Cylindrical warping (Affine Transformation)

output-homography-lpb.png: Result of Cylindrical warping (Affine Transformation) with Laplacian Blending.
