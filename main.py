# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Bonus perspective warping")
   print("4 Bonus cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
   
# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):
	
    # Write your codes here
    #im1 = cv2.imread("image1.jpg", 0)
    #im2 = cv2.imread("image2.jpg", 0)
    (x, y) = img1.shape
    
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask1) = getTransform(img3, img1,'homography')
    (M1, pts3, pts4, mask2) = getTransform(img2, img1,'homography')
    
    (x, y) = img1.shape
   	
    # for example: transform img2 to img1's plane
    # first, make some room around img1
    #out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    # then transform im1 with the 3x3 transformation matrix
    out1 = cv2.warpPerspective(img3, M, (img1.shape[1],img1.shape[0]))
    out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
    output = np.zeros(img1.shape)

    for i in range(x):
	for j in range(y):
		if img1[i][j]==0 and out1[i][j]==0:
			output[i][j]=0
    		elif img1[i][j]==0:
			output[i][j] = out1[i][j]
		elif out1[i][j]==0:
    			output[i][j] = (img1[i][j])
		else:
			output[i][j]= (int(int(img1[i][j]) + int(out1[i][j]))/2) 
	
    output1 = np.zeros(output.shape)
    	
    for i in range(x):
	for j in range(y):
		if output[i][j]==0 and out2[i][j]==0:
			output1[i][j]=0
    		elif output[i][j]==0:
			output1[i][j] = out2[i][j]
		elif out2[i][j]==0:
    			output1[i][j] = (output[i][j])
		else:
			output1[i][j]= (int(int(output[i][j]) + int(out2[i][j]))/2) 

    cv2.imwrite('output_homography.png',output1)
    o = cv2.imread('output_homography.png',0)	
    
    master = cv2.imread("example_output1.png", 0)	
    
    
    output_image = o 
    output_name = sys.argv[5] + "output_homography.png"
    cv2.imwrite(output_name, output_image)
    return True

def Laplacian_blending(img1,img2,mask,levels=4):
	
	G1 = img1.copy()
    	G2 = img2.copy()
    	GM = mask.copy()
    	gp1 = [G1]
   	gp2 = [G2]
    	gpM = [GM]
    	for i in xrange(levels):
        	G1 = cv2.pyrDown(G1)
        	G2 = cv2.pyrDown(G2)
        	GM = cv2.pyrDown(GM)
        	gp1.append(np.float32(G1))
        	gp2.append(np.float32(G2))
        	gpM.append(np.float32(GM))

    	# generate Laplacian Pyramids for A,B and masks
    	lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    	lp2  = [gp2[levels-1]]
    	gpMr = [gpM[levels-1]]
    	for i in xrange(levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        	L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
        	L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
        	lp1.append(L1)
        	lp2.append(L2)
        	gpMr.append(gpM[i-1]) # also reverse the masks

    	# Now blend images according to mask in each level
    	LS = []
    	for l1,l2,gm in zip(lp1,lp2,gpMr):
        	ls = l1 * gm + l2 * (1.0 - gm)
       		LS.append(ls)

    	# now reconstruct
    	ls_ = LS[0]
    	for i in xrange(1,levels):
        	ls_ = cv2.pyrUp(ls_)
		ls_ = cv2.add(ls_, LS[i])

	return ls_

def Bonus_perspective_warping(img1, img2, img3):
	
	img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    	(M, pts1, pts2, mask1) = getTransform(img3, img1,'homography')
        (M1, pts3, pts4, mask2) = getTransform(img2, img1,'homography')
    	
	m = np.ones_like(img3, dtype='float32')
	m1 = np.ones_like(img2, dtype='float32')
	
	
	out1 = cv2.warpPerspective(img3, M, (img1.shape[1],img1.shape[0]))
	out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
	out3 = cv2.warpPerspective(m, M, (img1.shape[1],img1.shape[0]))
	out4 = cv2.warpPerspective(m1, M1, (img1.shape[1],img1.shape[0]))
    	
	lpb = Laplacian_blending(out1,img1,out3,4)
	
	lpb1 = Laplacian_blending(out2,lpb,out4,4)
	cv2.imwrite('output_homography_lpb.png',lpb1)
	o=cv2.imread('output_homography_lpb.png',0)
	
	output_image = o 
	output_name = sys.argv[5] + "output_homography_lpb.png"
	cv2.imwrite(output_name, output_image)
	
	return True

# ===================================================
# =============== Cynlindrical Warping ==============
# ============	=======================================
def Cylindrical_warping(img1, img2, img3):
	
	# Write your codes here
	
	h,w = img1.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img1,mask1 = cylindricalWarpImage(img1, K)
	img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)	

	
	h,w = img2.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img2,mask2 = cylindricalWarpImage(img2, K)
	img2 = cv2.copyMakeBorder(img2,50,50,300,300, cv2.BORDER_CONSTANT)
   
	
	h,w = img3.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img3,mask3 = cylindricalWarpImage(img3, K)
	img3 = cv2.copyMakeBorder(img3,50,50,300,300, cv2.BORDER_CONSTANT)
             
	(M, pts1, pts2, mask5) = getTransform(img3, img1,'affine')
    	(M1, pts3, pts4, mask6) = getTransform(img2, img1,'affine')
   	
   	out1 = cv2.warpAffine(img3, M, (img1.shape[1],img1.shape[0]))
   	out2 = cv2.warpAffine(img2, M1, (img1.shape[1],img1.shape[0]))
   	output = np.zeros(img1.shape)
    	#output = [[0 for x in range(x)] for y in range(y)] 
	
	x,y = img1.shape
    	for i in range(x):
		for j in range(y):
			if img1[i][j]==0 and out1[i][j]==0:
				output[i][j]=0
    			elif img1[i][j]==0 and out1[i][j]!=0:
				output[i][j] = out1[i][j]
			elif out1[i][j]==0 and img1[i][j]!=0:
    				output[i][j] = (img1[i][j])
			else:
				output[i][j]= (int(int(img1[i][j]) + int(out1[i][j]))/2) 
	
    	output1 = np.zeros(output.shape)
    	
    	for i in range(x):
		for j in range(y):
			if output[i][j]==0 and out2[i][j]==0:
				output1[i][j]=0
    			elif output[i][j]==0 and out2[i][j]!=0:
				output1[i][j] = out2[i][j]
			elif out2[i][j]==0 and output[i][j]!=0:
    				output1[i][j] = output[i][j]
			else:
				output1[i][j]= (int(int(output[i][j]) + int(out2[i][j]))/2) 

    	cv2.imwrite('output_cylindrical.png',output1)
    	o = cv2.imread('output_cylindrical.png',0)	
    
    	output_image = o # This is dummy output, change it to your output
	
	# Write out the result
	output_name = sys.argv[5] + "output_cylindrical.png"
	cv2.imwrite(output_name, output_image)
	
	return True

def Bonus_cylindrical_warping(img1, img2, img3):
		
	m = np.ones_like(img3, dtype='float32')
	m1 = np.ones_like(img2, dtype='float32')

	h,w = m.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	m,maskA = cylindricalWarpImage(m, K)
	m = cv2.copyMakeBorder(m,50,50,300,300, cv2.BORDER_CONSTANT)	


	h,w = m1.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	m1,maskB = cylindricalWarpImage(m1, K)
	m1 = cv2.copyMakeBorder(m1,50,50,300,300, cv2.BORDER_CONSTANT)	


	h,w = img1.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img1,mask1 = cylindricalWarpImage(img1, K)
	img1 = cv2.copyMakeBorder(img1,50,50,300,300, cv2.BORDER_CONSTANT)	

	
	h,w = img2.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img2,mask2 = cylindricalWarpImage(img2, K)
	img2 = cv2.copyMakeBorder(img2,50,50,300,300, cv2.BORDER_CONSTANT)
   
	
	h,w = img3.shape
	f = 400
	K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
	img3,mask3 = cylindricalWarpImage(img3, K)
	img3 = cv2.copyMakeBorder(img3,50,50,300,300, cv2.BORDER_CONSTANT)
	
	(M, pts1, pts2, mask5) = getTransform(img3, img1,'affine')
    	(M1, pts3, pts4, mask6) = getTransform(img2, img1,'affine')
	
   	out1 = cv2.warpAffine(img3, M, (img1.shape[1],img1.shape[0]))
   	out2 = cv2.warpAffine(img2, M1, (img1.shape[1],img1.shape[0]))
	out3 = cv2.warpAffine(m, M, (img1.shape[1],img1.shape[0]))
	out4 = cv2.warpAffine(m1, M1, (img1.shape[1],img1.shape[0]))

	lpb = Laplacian_blending(out1,img1,out3,3)
	
	lpb1 = Laplacian_blending(out2,lpb,out4,3)

	cv2.imwrite('output_cylindrical_lpb.png',lpb1)
	o=cv2.imread('output_cylindrical_lpb.png',0)
	
	# Write your codes here
	output_image = o 
	output_name = sys.argv[5] + "output_cylindrical_lpb.png"
	cv2.imwrite(output_name, output_image)
	
	return True
	
'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''
def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)

        return total_diff;
if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) != 6):
      help_message()
      sys.exit()
   else: 
      question_number = int(sys.argv[1])
      if (question_number > 4 or question_number < 1):
	 print("Input parameters out of bound ...")
         sys.exit()
		 
   input_image1 = cv2.imread(sys.argv[2], 0)
   input_image2 = cv2.imread(sys.argv[3], 0)
   input_image3 = cv2.imread(sys.argv[4], 0) 

   function_launch = {
   1 : Perspective_warping,
   2 : Cylindrical_warping,
   3 : Bonus_perspective_warping,
   4 : Bonus_cylindrical_warping
   }

   # Call the function
   function_launch[question_number](input_image1, input_image2, input_image3)
