from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops
from skimage import measure
import math
from scipy import stats
import numpy as np
from scipy.integrate import quad

# Hyper-parameters For algorithm
# For spatial Cue
Variance = 3
# K-means cluster
k = 10
# Input Image
src = "car2.jpg"
name = "car2"
nPx = 150

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def centre(x,sigma):
    C = sigma * math.sqrt(2*math.pi)
    e = math.exp((-1*(dist))/(2*(sigma**2)))  
    ans = e/C
    return ans

def mapper(a1,a2,b1,b2,s):
    t = b1 + (((s-a1)*(b2-b1))/(a2-a1))
    return t

spx = {}
kpx = {}
###
centers=[]
label=[]
# Constrast Cue
cc = []
dim = [] # to store(h,w)
image = cv2.imread(src)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imagec = cv2.imread(src,0) 
images = cv2.imread(src,0)  
imgc= cv2.imread(src,0)
imgs= cv2.imread(src,0)
sal= cv2.imread(src,0)
h,w = image.shape[:2]
dim.append((h,w))
imcount = 1

# 2d array (same size as of image) with unique nos. to pixels representing cluster ids
segments = slic(image, n_segments = nPx,start_label = 1, compactness = 10)
fig = plt.figure("Superpixels -- %d segments" % (300))
plt.imshow(mark_boundaries(image, segments))
plt.savefig("SuperPixel.png")

# Kmeans
pixel_val = []
areas = regionprops(segments,intensity_image=image)
for cluster in areas:
    no = cluster.label
    mean = cluster.mean_intensity
    conv = np.float32((mean[0],mean[1],mean[2]))
    spx[(conv[0],conv[1],conv[2])] = no   
    pixel_val.append([mean[0],mean[1],mean[2]])
    
pixel_val = np.array(pixel_val)    
pixel_val = np.float32(pixel_val)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
_, labels, (center) = cv2.kmeans(pixel_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)   # [[],[],[]...k centres]
labels = labels.flatten()       #[x,x,,x,x,y,y,.....(h*w) pixels label]flatten [[],[],[]...[]] -> [_,_,_,_,_,_]
maxx = max(labels)
for i in range(1,maxx+2):
    tt = []
    for j in range(len(labels)):
        if ((labels[j]+1) == i):
            tt.append(list(pixel_val[j]))
    kpx[i] = tt    
nlabels = []
for i in range(h):
    nlabels.append([])
    for j in range(w):
        nlabels[-1].append(0)

for i in kpx.keys():   # i = new id
    for j in kpx[i]:
        val = spx[tuple(j)]  # segments ids
        for x in range(h):
            for y in range(w):
                if (segments[x][y] == val):
                    nlabels[x][y] = i

nlabels  = np.array(nlabels,dtype='int32')     # nlabels is the new segments
fig = plt.figure("Superpixels -- %d segments" % (300))
plt.imshow(mark_boundaries(image, nlabels))
plt.savefig("kmeans.png")
regions = regionprops(nlabels,intensity_image=image)
WK = []
for props in regions:  # K
    cx, cy = props.coords[0]
    Kmean = props.mean_intensity
    centers.append((cx,cy))   # first coordinate
    # has cluster no. of centres of clusters
    label.append(nlabels[cx][cy]) 

    ### Contrast Cue ###
    summ = 0
    for reg in regions: # i
        px = reg.area
        weight = px / (h*w) ##
        imean = reg.mean_intensity   # RGB mean
        temp = (Kmean-imean)**2 ##   # array of 3
        dist = math.sqrt(temp[0]+temp[1]+temp[2])
        summ += (weight * dist)
    imgc[cx,cy]= summ

    ### Spatial Cue ###
    nk = 1 / (props.area) ##
    plus = 0
    for i in range(imcount):  #here image is one
        #h,w = dim[i]
        for x in range(h): # no. of pixel
            for y in range(w):
                if (nlabels[x][y] == props.label):  # or label[-1]
                    dist = math.sqrt( ((x-(h/2))**2) + ((y-(w/2))**2) )
                    plus += centre(dist,Variance)
    wk = nk * plus
    WK.append(wk)
    
maxwk = max(WK)
minwk = min(WK)
for i in range(len(centers)):
    val = mapper(minwk,maxwk,0,255,WK[i])
    imgs[centers[i][0],centers[i][1]] = val
    print(WK[i],val

print ("Superpixels genreated....")

########   CUES   ##################
for i in range(h):
        for j in range(w):
                lb=nlabels[i][j]
                for t in range(len(label)):
                        if (label[t]==lb):
                                imagec[i,j]=imgc[centers[t][0],centers[t][1]]
                                break
cv2.imwrite('Contrast_'+'px'+str(nPx)+name+'_'+str(k)+'_'+str(Variance)+'.png',imagec)
test = [0]
for i in range(h):
        for j in range(w):
                lb=nlabels[i][j]
                for t in range(len(label)):
                	  if (label[t]==lb):
                       images[i,j]=imgs[centers[t][0],centers[t][1]]
                                break
                            
cv2.imwrite('Spatial_'+'px'+str(nPx)+name+'_'+str(k)+'_'+str(Variance)+'.png',images)
print("doing saliency")

def intersection(uf,ub,sf,sb):
    # Equation 5 in Research Paper
    ans1 = ((ub*(sf**2)) - (uf*(sb**2))) / ((sf**2) - (sb**2))
    ans2 = ((sf*sb) / ((sf**2)  - (sb**2)))
    ans3 = math.sqrt(( (uf-ub)**2 ) - ( 2*((sf**2) - (sb**2)) * (math.log(sb)-math.log(sf)) ))
    return (ans1+(ans2*ans3)),(ans1-(ans2*ans3))

def Df(z,ub,uf,sf,sb):
    val = math.exp(-1*(((z-uf)/sf)**2)) / (sf*math.sqrt(2*math.pi))
    return val

def Db(z,ub,uf,sf,sb):
    val = math.exp(-1*(((z-ub)/sb)**2)) / (sb*math.sqrt(2*math.pi))
    return val
    
def sep_measure(th,pic):
    h,w = pic.shape[:2]
    Fg = []
    Bg = []
    for i in range(h):
        for j in range(w):
            if (pic[i,j] < th):
                Bg.append(pic[i,j]/255)
            else:
                Fg.append(pic[i,j]/255)
    # mean            
    uf = sum(Fg)/len(Fg)
    ub = sum(Bg)/len(Bg)
    print("mean",uf,ub)
    # variance
    varf = 0
    varb = 0
    for i in range(len(Fg)):
        varf += ((Fg[i]-uf)**2)
    for i in range(len(Bg)):
        varb += ((Bg[i]-ub)**2)
    print("variance",varf,varb)    
    sigf = math.sqrt(varf / len(Fg))
    sigb = math.sqrt(varb / len(Bg))
    print("final variance",sigf,sigb)
    # finding z*
    z_inter = max(intersection(uf,ub,sigf,sigb))
    print("z_star",z_inter)
    # Df, Db Integration
    I_Df = quad(Df,0,z_inter,args=(ub,uf,sigf,sigb))[0]
    I_Db = (quad(Db,z_inter,1,args=(ub,uf,sigf,sigb))[0])
    print("Integrals",I_Df,I_Db)
    Bin = len(Fg)+len(Bg)
    return 1/(1+math.log10(1+Bin*(I_Df+I_Db)))
# Contrast
thres1, pic1 = cv2.threshold(imagec,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("t1",thres1)
w1 = sep_measure(thres1,imagec)
print("Contrast Score",w1)
# Spatial
thres2, pic2 = cv2.threshold(images,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("t2",thres2)
w2 = sep_measure(thres2,images)
print("Spatial Score",w2)
h,w = images.shape[:2]
for i in range(h):
    for j in range(w):
        sal[i,j] = (w1*int(imagec[i,j])) + (w2*int(images[i,j]))

# Final Saliency Map
cv2.imwrite('Saliency_'+'px'+str(nPx)+name+'_'+str(k)+'_'+str(Variance)+'.png',sal)
