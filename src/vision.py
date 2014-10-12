%matplotlib inline
import numpy as np
import cv2
from matplotlib import pyplot as plt
import inspect

DEBUGPLOT = 1
def dplt(image):
    if not DEBUGPLOT: return
    print inspect.stack()[1]
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

import cv2
import numpy as np  
import numpy.linalg as lin 

# todo: clean up data structures
# i think right now there are too many of them

# todo: 
# parametrize: GBkernel, SEkernel
def preprocess(image):
    gbkernel = 9
    blur = cv2.GaussianBlur(image,(gbkernel,gbkernel),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    #div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(gray,gray,0,255,cv2.NORM_MINMAX))

    dplt(res)
    return res  

# todo:
# what needs to be parametrized here?
# maybe tidy up datatypes (e.g. corners)?
# sort Corners lexicographically + update transformView accordingly
def findGobanCorners(image):
    mask = np.zeros((image.shape),np.uint8)
    thresh = cv2.adaptiveThreshold(image,255,0,1,19,2)
    dplt(thresh) # threshold

    contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = max(contour, key=lambda c: cv2.contourArea(c))

    cv2.drawContours(mask,[best_cnt],0,255,-1)
    #cv2.drawContours(mask,[best_cnt],0,0,2)q
    dplt(mask) # contours
    res = cv2.bitwise_and(image,mask)

    edges = cv2.Canny(res, 50, 100)
    dplt(edges) # canny
    lines = cv2.HoughLines(edges, 1, np.pi/(360), 200)
    lines = lines[0]
    m,n = mask.shape

    Z = np.float32(lines)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,4, (cv2.TERM_CRITERIA_EPS, 30, 0.1), 10, 0)

    rs = []
    ts = []
    for n in range(0,4):
        rs.append(center[n:(n+1)][0][0])
        ts.append(center[n:(n+1)][0][1])

    corners = []
    height, width = image.shape[:2] 
    for i in range(0,4):
        for j in range(i,4):
            if i != j:
                mat = np.matrix([[np.cos(ts[i]), np.sin(ts[i])],[np.cos(ts[j]), np.sin(ts[j])]])
                rhs = np.matrix([[rs[i]],[rs[j]]])
                cornerCandidate = np.linalg.solve(mat,rhs)
                if (cornerCandidate[0] > 0) and (cornerCandidate[0] < width) and (cornerCandidate[1] > 0) and (cornerCandidate[1] < height):
                    corners.append(cornerCandidate)

    """
    # Plot borders and corners
    image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
    for (rho, theta) in center:
        x0 = np.cos(theta)*rho 
        y0 = np.sin(theta)*rho
        pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
        pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
        cv2.line(image, pt1, pt2, (0,0,255), 10)  

    for n in range(0,4):
        cv2.circle(image,(corners[n][0], corners[n][1]),20,(255,0,0),10)
    """

    return corners 


# todo:
# parametrize: output image size 
# tidy up matrix datatypes?
# apply sorting from findGobanCorners
def transformView(image, corners):
    M_img = np.matrix([ 
                [float(corners[0][0]),  float(corners[0][1])],
                [float(corners[1][0]),  float(corners[1][1])],
                [float(corners[2][0]),  float(corners[2][1])],
                [float(corners[3][0]),  float(corners[3][1])] ])
                            
    M_new = np.matrix([ 
                [950.0, 50],
                [50,    50],
                [950,  950],
                [50,   950]  ])

    H = cv2.findHomography(M_img,M_new)
    res = cv2.warpPerspective(image,H[0],(1000,1000)) 
     
    return res


# todo
def detectIntersections(image):
    closex = verticalClosing(image)
    closey = horizontalClosing(image)
    res = cv2.bitwise_and(closex,closey)
      
    image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    intersections = []
    n = 1
    for cnt in contour:
        mom = cv2.moments(cnt)
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        intersections.append((x,y))
        cv2.putText(image, str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
        n = n+1
        cv2.circle(image,(x,y),6,(0,0,255),-1)

    return intersections

# todo
# data structures, kmeans parametrization
def computeIntersections(intersections):
    xs = []
    ys = []
    for (x,y) in intersections:
            xs.append(x)
            ys.append(y)  

    xs = np.float32(xs)
    ys = np.float32(ys)

    retX,labelX,centerX=cv2.kmeans(xs,19, (cv2.TERM_CRITERIA_EPS, 30, 0.1), 30, 0)
    retY,labelY,centerY=cv2.kmeans(ys,19, (cv2.TERM_CRITERIA_EPS, 30, 0.1), 30, 0)

    centerX = sorted([item for sublist in centerX for item in sublist])
    centerY = sorted([item for sublist in centerY for item in sublist])

    computedIntersections = []
    for y in centerY:
        for x in centerX:
           computedIntersections.append((x,y)) 

    return computedIntersections 


# todo
# parametrize SE-shape, morphology iterations, sobel aperture
#               maybe thresholds?
def verticalClosing(image):
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,5))
    dx = cv2.Sobel(image,cv2.CV_16S,1,0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
       
    return close 


# todo
# parametrize SE-shape, morphology iterations, sobel aperture
#               maybe thresholds?
# maybe reuse verticalClosing after parametrizing instead of defining new function?
def horizontalClosing(image):
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
    dy = cv2.Sobel(image,cv2.CV_16S,0,1)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely,iterations = 1)

    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w/h > 5:
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)

    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)

    return close


# todo
# parametrize and think how to
# make test more elaborate, maybe second threshold?
# couple stonesize with output size parametrization from transformView
# think about output datastructure
def scanGoban(image, intersections):
    #blur = cv2.GaussianBlur(image,(7,7),0)
    null, th = cv2.threshold(image,45,256,cv2.THRESH_BINARY)

    position = [] 
    for i in intersections:
        mask = np.zeros(image.shape, np.uint8)
        cv2.circle(mask,(i[0],i[1]),20,(255,255,255),-1)
        th_mask = cv2.bitwise_and(th,mask)

        hist_mask = cv2.calcHist([th_mask],[0],mask,[256],[0,256])
        cv2.normalize(hist_mask,hist_mask,1)

        sumBlack = sum(hist_mask[0:15])
        sumWhite = sum(hist_mask[240:256])

        if (sumBlack > 0.8) and (sumWhite < 0.08):
            cv2.circle(image,(i[0],i[1]),10,(0,0,0),-1) 
            cv2.circle(image,(i[0],i[1]),13,(255,0,0),3) 
        elif (sumWhite > 0.8) and (sumBlack < 0.08):
            cv2.circle(image,(i[0],i[1]),10,(255,0,0),-1) 
            cv2.circle(image,(i[0],i[1]),13,(0,0,0),3) 

    return image

def test():
    img = cv2.imread('gob1.jpg',1)
    pp = preprocess(img)
    corners = findGobanCorners(pp)
    tv = transformView(img, corners)
    dplt(tv)