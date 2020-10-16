#https://github.com/sudo-sherry/
#coded by Shaharyar
#version 1.0.1 beta test
import numpy as np
import cv2
import matplotlib.pyplot as pt
img = cv2.imread('c.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,10,3,0.04)
ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#pt.imshow(img)
pts = corners
pts = pts[1:]
for corner in pts:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
import math
origin = pts[0]
refvec = [0, 1]
def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector
pts = sorted(pts, key=clockwiseangle_and_distance)
pts = np.array(pts,dtype=int)
pts = np.array(pts).tolist()
pt.imshow(img,origin=(0,0))
pt.show()
#cv2.WINDOW_AUTOSIZE()
#namedWindow(“”, WINDOW_NORMAL) 
#cv2.imshow('Corner',img)