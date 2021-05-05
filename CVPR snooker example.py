#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math, cv2, time
from matplotlib import pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap

A = cv2.imread('./data/frame.png')
w = A.shape[0]
h = A.shape[1]
plt.figure(figsize=(4, 3), dpi=120, facecolor='white')
plt.imshow(cv2.cvtColor(A,cv2.COLOR_BGR2RGB))
plt.show()


## Crop image to snooker table boundaries
hsv = cv2.cvtColor(A,cv2.COLOR_BGR2HSV)
H = hsv[:,:,0].astype(np.float)
# isolate green
H = np.zeros((w,h))
H[np.logical_and(hsv[:,:,0]> 20,hsv[:,:,0] < 90)] = 1
#plt.imshow(H)
#plt.viridis()
#plt.colorbar()
#plt.show()

W = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
Gx = cv2.filter2D(H,-1,W,cv2.BORDER_REPLICATE)
Gy = cv2.filter2D(H,-1,W.T,cv2.BORDER_REPLICATE)
G = np.abs(Gx)+np.abs(Gy)
#plt.imshow(G)
#plt.show()

# Simple hough space along x and y directions
Ax = np.sum(G,axis=1)
Ay = np.sum(G,axis=0)

# non-maxima suppression
d = 10
Ix = Ax.copy()
for i in range(d,Ix.shape[0]-d):
    A_d = Ix[i-d:i+d+1].copy()
    A_d[d] = 0
    if Ix[i] <= np.amax(A_d):
        Ix[i] = 0

Iy = Ay.copy()
for i in range(d,Iy.shape[0]-d):
    A_d = Iy[i-d:i+d+1].copy()
    A_d[d] = 0
    if Iy[i] <= np.amax(A_d):
        Iy[i] = 0

# get two central peaks
Px = np.sort(np.argsort(-Ix)[:4])
Px = Px[1:3]
Py = np.sort(np.argsort(-Iy)[:4])
Py = Py[1:3]

plt.figure(figsize=(4, 3), dpi=120, facecolor='white')
plt.plot(Ix)
plt.plot(Px, Ix[Px], "x")
plt.title('width')
plt.show()

plt.figure(figsize=(4, 3), dpi=120, facecolor='white')
plt.plot(Iy)
plt.plot(Py, Iy[Py], "x")
plt.title('length')
plt.show()

Px[0] += 13
Px[0] += 5
Px[1] -= 16

Py[0] += 15
Py[1] -= 16

print('Peak locations:',Px,Py)
plt.figure(figsize=(4, 3), dpi=120)
plt.imshow(cv2.cvtColor(A[Px[0]:Px[1],Py[0]:Py[1],:],cv2.COLOR_BGR2RGB))
plt.show()

mask = np.zeros((w,h))
mask[Px[0]:Px[1],Py[0]:Py[1]] = 1

d = 20
mask[Px[0]-d:Px[0]+d,Py[0]-d:Py[0]+d] = 0
mask[Px[1]-d:Px[1]+d,Py[0]-d:Py[0]+d] = 0
mask[Px[0]-d:Px[0]+d,Py[1]-d:Py[1]+d] = 0
mask[Px[1]-d:Px[1]+d,Py[1]-d:Py[1]+d] = 0

plt.figure(figsize=(4, 3), dpi=120, facecolor='white')
plt.imshow(mask)
plt.viridis()
plt.colorbar()
plt.show()


## Compute ball colour segmentation of image
cspace = cv2.COLOR_BGR2LAB # [cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB]
bg = cv2.cvtColor(cv2.imread('./data/bg2.png'),cspace).reshape(-1,3) # background green
r = cv2.cvtColor(cv2.imread('./data/red.png'),cspace).reshape(-1,3) # red ball
g = cv2.cvtColor(cv2.imread('./data/green.png'),cspace).reshape(-1,3) # green ball
b = cv2.cvtColor(cv2.imread('./data/blue.png'),cspace).reshape(-1,3) # blue ball
k = cv2.cvtColor(cv2.imread('./data/black.png'),cspace).reshape(-1,3) # black ball
m = cv2.cvtColor(cv2.imread('./data/brown.png'),cspace).reshape(-1,3) # brown ball
p = cv2.cvtColor(cv2.imread('./data/pink.png'),cspace).reshape(-1,3) # pink ball
w = cv2.cvtColor(cv2.imread('./data/white.png'),cspace).reshape(-1,3) # white ball
y = cv2.cvtColor(cv2.imread('./data/yellow.png'),cspace).reshape(-1,3) # yellow ball
print('loading done')

X = np.concatenate((bg,r,g,b,k,m,p,w,y))
y = np.concatenate((0*np.ones(bg.shape[0]),1*np.ones(r.shape[0]),2*np.ones(g.shape[0]),3*np.ones(b.shape[0]),4*np.ones(k.shape[0]),5*np.ones(m.shape[0]),6*np.ones(p.shape[0]),7*np.ones(w.shape[0]),8*np.ones(y.shape[0])))

print('fitting model...',end='')
#clf = svm.LinearSVC(max_iter=100000)
clf = svm.SVC()
clf.fit(X, y)
print('done.')
yPred = clf.predict(cv2.cvtColor(A,cspace).reshape(-1,3))
yPred = yPred.reshape(A.shape[0],A.shape[1],1)

plt.figure(figsize=(8, 6), dpi=120)
textmap = ['bg','red','green','blue','black','brown','pink','white','yellow']
mymap = [[0,0.5,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1],[0.65,0.15,0.15,1],[1,0.75,0.8,1],[1,1,1,1],[1,1,0,1]]
plt.imshow(np.multiply(yPred,mask.reshape(mask.shape[0],mask.shape[1],1)),cmap=ListedColormap(mymap))
plt.colorbar()
#plt.axis('off')
plt.show()


## Compute background/foreground segmentation
bg_cspace = cv2.COLOR_BGR2HSV # [cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB]
bg = cv2.cvtColor(cv2.imread('./data/bg.png'),bg_cspace).reshape(-1,3) # background green
r = cv2.cvtColor(cv2.imread('./data/red.png'),bg_cspace).reshape(-1,3) # red ball
g = cv2.cvtColor(cv2.imread('./data/green.png'),bg_cspace).reshape(-1,3) # green ball
b = cv2.cvtColor(cv2.imread('./data/blue.png'),bg_cspace).reshape(-1,3) # blue ball
k = cv2.cvtColor(cv2.imread('./data/black.png'),bg_cspace).reshape(-1,3) # black ball
m = cv2.cvtColor(cv2.imread('./data/brown.png'),bg_cspace).reshape(-1,3) # brown ball
p = cv2.cvtColor(cv2.imread('./data/pink.png'),bg_cspace).reshape(-1,3) # pink ball
w = cv2.cvtColor(cv2.imread('./data/white.png'),bg_cspace).reshape(-1,3) # white ball
y = cv2.cvtColor(cv2.imread('./data/yellow.png'),bg_cspace).reshape(-1,3) # yellow ball
print('loading done')

fg = np.concatenate((r,g,b,k,m,p,w,y))
X = np.concatenate((bg,fg))
y = np.concatenate((0*np.ones(bg.shape[0]),1*np.ones(fg.shape[0])))

print('fitting model...',end='')
bgClf = svm.SVC()
bgClf.fit(X, y)
print('done.')

bgPred = bgClf.predict(cv2.cvtColor(A,bg_cspace).reshape(-1,3))
bgPred = bgPred.reshape(A.shape[0],A.shape[1],1)

plt.figure(figsize=(8, 6), dpi=120)
textmap2 = ['bg','fg']
mymap2 = [[0,1,0,1],[1,0,0,1]]
plt.imshow(bgPred,cmap=ListedColormap(mymap2))
#plt.colorbar()
#plt.axis('off')
plt.show()



## Hough circle transform
def find_circles(I,mask,r=6,thresh=1500):
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(I)
    plt.colorbar()
    plt.title('input')
    plt.show()
    
    w = I.shape[0]
    h = I.shape[1]
    
    W = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Gx = cv2.filter2D(I,-1,W,cv2.BORDER_REPLICATE)
    Gy = cv2.filter2D(I,-1,W.T,cv2.BORDER_REPLICATE)
    Gx = np.multiply(Gx,mask)
    Gy = np.multiply(Gy,mask)
    G = np.abs(Gx)+np.abs(Gy)
    
    Go = np.arctan(np.divide(Gy,Gx+0.00000000001))
    
    X,Y = np.meshgrid(range(0,w), range(0,h), sparse=False, indexing='ij')
    A = X - r*np.cos(Go[X,Y]) + r
    B = Y - r*np.sin(Go[X,Y]) + r
    A = np.round(A).astype(np.int)
    B = np.round(B).astype(np.int)
    
    H = np.zeros((w+r*2,h+r*2))
    np.add.at(H,(A,B),G[X,Y])
    
    #H = cv2.erode(H,np.ones((3,3),np.uint8),iterations = 1)
    
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(np.sqrt(H))
    plt.colorbar()
    #plt.axis('off')
    plt.show()
    
    # non-maxima suppression
    d = r*1
    for i in range(d,w-d):
        for j in range(d,h-d):
            H_d = H[i-d:i+d+1,j-d:j+d+1].copy()
            H_d[d,d] = 0
            #H2[i,j] = np.amax(H_d)
            if H[i,j] <= np.amax(H_d):
                H[i,j] = 0
    
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(np.sqrt(H))
    plt.colorbar()
    #plt.axis('off')
    plt.show()
    
    print('No. of circles found:',np.count_nonzero(H[H > thresh]))
    
    markers = np.argwhere(H > thresh) - r
    scores = H[H > thresh] / (r*r*255)
    return markers, scores


# Find location of balls
markers, scores = find_circles(np.multiply(bgPred.reshape(A.shape[0],A.shape[1])*255, mask),mask)

# Find colour of balls
colours = np.array([int(np.median(clf.predict(cv2.cvtColor(A[marker[0]-3:marker[0]+3,marker[1]-3:marker[1]+3,:],cspace).reshape(-1,3)))) for marker in markers])

for i in range(2,np.amax(colours)): # skip red and background (bg)
    if np.count_nonzero(colours == i) > 1:
        conflicts = np.arange(colours.size)[colours == i]
        conflicts = np.delete(conflicts,np.argmax(scores[conflicts]))
        markers = np.delete(markers,conflicts,axis=0)
        scores = np.delete(scores,conflicts)
        colours = np.delete(colours,conflicts)

# Draw result
plt.figure(figsize=(16, 12), dpi=80)
plt.imshow(cv2.cvtColor(A,cv2.COLOR_BGR2RGB))
for i in range(0,markers.shape[0]):
    marker = markers[i]
    colour = colours[i]
    if colour > 0 and bgPred[marker[0],marker[1]] > 0:
        print(marker,colour,textmap[colour], scores[i])
        plt.scatter(marker[1], marker[0],50,[mymap[colour]])
        plt.text(marker[1], marker[0],textmap[colour] + ' %.2f' % scores[i])

plt.show()

print('done!')
import sys
sys.exit()
## Compute video
cap = cv2.VideoCapture('./scottish-open-2020-osullivan-snippet.mp4')

if cap.isOpened():
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,fps,(w,h))
    
    it = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        it += 1
        
        if not it % 10 == 0:
            continue
        
        if it > 100:
            break
        
        print('processing frame:',it)
        
        A = frame
        
        # find background/balls
        print('segmenting background...',end='')
        t0 = time.time()
        bgPred = bgClf.predict(cv2.cvtColor(A,bg_cspace).reshape(-1,3))
        bgPred = bgPred.reshape(A.shape[0],A.shape[1],1)
        print('done. %.2fs' % (time.time()-t0))
        
        print('finding circles...',end='')
        t0 = time.time()
        markers, scores = find_circles(np.multiply(bgPred.reshape(A.shape[0],A.shape[1])*255, mask),mask)
        colours = np.array([int(np.median(clf.predict(cv2.cvtColor(A[marker[0]-3:marker[0]+3,marker[1]-3:marker[1]+3,:],cspace).reshape(-1,3)))) for marker in markers])
        print('done. %.2fs' % (time.time()-t0))

        print('removing duplicates...',end='')
        t0 = time.time()
        for i in range(2,np.amax(colours)): # skips red and background (bg)
            if np.count_nonzero(colours == i) > 1:
                conflicts = np.arange(colours.size)[colours == i]
                conflicts = np.delete(conflicts,np.argmax(scores[conflicts]))
                markers = np.delete(markers,conflicts,axis=0)
                scores = np.delete(scores,conflicts)
                colours = np.delete(colours,conflicts)
        print('done. %.2fs' % (time.time()-t0))

        print('draw results')
        plt.figure(figsize=(16, 12), dpi=80)
        plt.imshow(cv2.cvtColor(A,cv2.COLOR_BGR2RGB))
        for i in range(0,markers.shape[0]):
            marker = markers[i]
            colour = colours[i]
            if colour > 0 and bgPred[marker[0],marker[1]] > 0:
                print(marker,colour,textmap[colour], scores[i])
                plt.scatter(marker[1], marker[0],50,[mymap[colour]])
                plt.text(marker[1], marker[0],textmap[colour] + ' %.2f' % scores[i])
                cv2.circle(A, (marker[1], marker[0]), 6, (mymap[colour][2]*255,mymap[colour][1]*255,mymap[colour][0]*255), 2)

        plt.show()
        
        out.write(A)
        
    out.release()

cap.release()

