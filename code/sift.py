import cv2
import numpy as np
import random

# Load the two images to match with
img1 = cv2.imread('image_2.jpg')
img2 = cv2.imread('image_5.jpg')
print 'Loaded images..'

# Store the dimensions of images
height1, width1, depth1 = img1.shape
height2, width2, depth2 = img2.shape

# Stack two images together for final display
resimg2  = np.zeros((height1, width2, depth2),dtype='uint8')
resimg2[:height2,:width2, :] = img2
both = np.hstack((img1, resimg2))

# Detect SIFT features of the two images
sift_detector  = cv2.xfeatures2d.SIFT_create()
kp1, features1 = sift_detector.detectAndCompute(img1, None)
kp2, features2 = sift_detector.detectAndCompute(img2, None)
print 'SIFT features detected..'

numkp1 = len(kp1)
matches = []

# Find similar features b/w images and filter the matches using a threshold value
# Eucledian distance used for calculating similarity of SIFT features.
for i in range(numkp1):
    diffs = np.subtract(features2, features1[i])
    norms = np.linalg.norm(diffs, axis=1)
    sortednormidx = np.argsort(norms)
    mintwodiff = norms[sortednormidx[1]] - norms[sortednormidx[0]]

    if(mintwodiff >= 150):
        matches.append((kp1[i], kp2[sortednormidx[0]]))


print 'Found SIFT matches b/w two images..'


# RANSAC for finding Homography
N=1000
inliertreshold = 0.99

print 'Finding Homography..'

for i in range(N):
    print 'RANSAC iteration ' + str(i)

    # Solve Homography Matrix using 4 points
    rand3 = random.sample(matches, 4)

    a = np.array([ [match[0].pt[0] , match[0].pt[1] , 1 ] for match in rand3])
    b = np.array([ [match[1].pt[0] , match[1].pt[1] , 1 ] for match in rand3])

    H = cv2.solve(a, b, flags=cv2.DECOMP_SVD)
    H = H[1].T

    # Check if fraction of inliers is higher than treshold.
    fit = []
    fittreshold = 4
    for match in matches:
        x = [match[0].pt[0], match[0].pt[1], 1]
        y = [match[1].pt[0], match[1].pt[1], 1]

        result = np.dot(H, x)
        fit.append(np.linalg.norm(result-y))

    inliers = [f for f in fit if f < fittreshold]

    fraction = float(len(inliers))/len(fit)
    if(fraction>inliertreshold):
        break


print 'Homography H: '
print H

# Draw lines joining matching points
for match in matches:
    img1_kp = match[0]
    img2_kp = match[1]

    x1 = int(img1_kp.pt[0])
    y1 = int(img1_kp.pt[1])

    x2 = int(img2_kp.pt[0] + width1)
    y2 = int(img2_kp.pt[1])

    cv2.circle(both, (x1, y1), radius=1, color=(0, 0, 255), thickness=1);
    cv2.circle(both, (x2, y2), radius=1, color=(0, 0, 255), thickness=1);
    cv2.line(both, (x1, y1), (x2, y2), color=(0, 0, 255))

# Draw circles on transformed SIFT feature points
for match in matches:
    X = np.array([ match[0].pt[0] , match[0].pt[1] , 1 ])
    X2 = np.array([ match[1].pt[0] , match[1].pt[1] , 1 ])
    Y = np.dot(H, X)
    cv2.circle(img2, (int(X2[0]), int(X2[1])), radius=2, color=(255, 0, 0), thickness=2);
    cv2.circle(img2, (int(Y[0]), int(Y[1])), radius=2, color=(0, 255, 0), thickness=2);

# Display images and write to file
cv2.namedWindow('b.Features Overlay',cv2.WINDOW_NORMAL)
cv2.imshow('b.Features Overlay', img2)
cv2.imwrite('img2_5_partb.jpg',img2.astype('uint8'))

cv2.namedWindow('a. SIFT matches',cv2.WINDOW_NORMAL)
cv2.imshow('a. SIFT matches', both)
cv2.imwrite('img2_5_parta.jpg', both.astype('uint8'))

cv2.waitKey(0)
cv2.destroyAllWindows()
