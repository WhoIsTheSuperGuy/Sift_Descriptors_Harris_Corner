import numpy as np
import cv2
import math
from scipy.ndimage import maximum_filter
np.seterr(divide='ignore', invalid='ignore')



def harrisCornerDetection(imgo):
	imgraw=input("Second Input\n")
	secondImg1=cv2.imread(imgraw)
	img=np.float32(imgo)
	secondImg=np.float32(secondImg1)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	secondImg=cv2.cvtColor(secondImg,cv2.COLOR_BGR2GRAY)
	height=img.shape[0]
	width=img.shape[1]
	secondHeight=secondImg.shape[0]
	secondwidth=secondImg.shape[1]
	hog=cv2.HOGDescriptor()
	dy,dx=np.gradient(img)
	dy2,dx2=np.gradient(secondImg)
	Ixx2=dx2**2
	Iyy2=dy2**2
	Ixy2=dx2*dy2
	Ixx2=cv2.GaussianBlur(Ixx2,(3,3),2)
	Iyy2=cv2.GaussianBlur(Iyy2,(3,3),2)
	Ixy2=cv2.GaussianBlur(Ixy2,(3,3),2)
	Ixx=dx**2
	Ixx=cv2.GaussianBlur(Ixx,(3,3),2)
	Iyy=dy**2
	Iyy=cv2.GaussianBlur(Iyy,(3,3),2)
	Ixy=dx*dy
	Ixy=cv2.GaussianBlur(Ixy,(3,3),2)
	thres=0.3
	windowsize=3
	rmax=0
	color_img=imgo.copy()
	#print(color_img.shape)
	color_img2=secondImg1.copy()
	#print(color_img2.shape)
	hog_image=imgo.copy()
	#finalimage=np.zeros((height,width))
	det= (Ixx*Iyy)-(Ixy**2)
	trace= Ixx+Iyy
	R=det/trace
	map(max, R)
	list(map(max, R))
	maxim=max(map(max, R))
	localmax=maximum_filter(R,size=3)
	localFilter=R ==localmax
	R=R * localFilter
	det1= (Ixx2*Iyy2)-(Ixy2**2)
	trace1= Ixx2+Iyy2
	R1=det1/trace1
	map(max, R1)
	list(map(max, R1))
	maxim1=max(map(max, R1))
	localmax1=maximum_filter(R1,size=3)
	localFilter1=R1 ==localmax1
	R1=R1 * localFilter1
	tempTuple=[]
	for y in range(0,height,2):
		for x in range(0,width,2):
			if(R[y,x]>(maxim*thres)):
				if x-8<0 or x+8>img.shape[0] or y-8<0 or y+8>img.shape[1]:
					continue;
				#finalimage[y,x]=r[y,x]
				tempTuple.append((y,x))
				color_img.itemset((y,x,0),0)
				color_img.itemset((y,x,1),0)
				color_img.itemset((y,x,2),255)
				cv2.rectangle(color_img,(x-2,y-2),(x+2,y+2),255,1)
	
	
	
	vis = np.concatenate((imgo, color_img), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	
	tempTuple1=[]
	for y in range(0,secondHeight,2):
		for x in range(0,secondwidth,2):
			if(R1[y,x]>(maxim1*thres)):
				#finalimage[y,x]=r[y,x]
				if x-8<0 or x+8>imgo.shape[0] or y-8<0 or y+8>imgo.shape[1]:
					continue;
				tempTuple1.append((y,x))
				color_img2.itemset((y,x,0),0)
				color_img2.itemset((y,x,1),0)
				color_img2.itemset((y,x,2),255)
				cv2.rectangle(color_img2,(x-2,y-2),(x+2,y+2),255,1)
	
	desDict = {}
	desDict1 = {}
	
	#secondImg=np.uint8(secondImg)	
	#hist1=hog.compute(secondImg,hog.blockStride,hog.cellSize,tempTuple1)
	#print(hist1.shape)
	arr = img.astype(np.float64)
	arr2=secondImg.astype(np.float64)
	for q in tempTuple:		
		desDict[(q[0],q[1])]=getdescriptor(arr,q[0],q[1])
		
	for e in tempTuple1:
		desDict1[(e[0],e[1])]=getdescriptor(arr2,e[0],e[1])
		
	#pdes=[]
	#ddes=[]
	#pdes=desDict
	#ddes=desDict1
	#print(type(pdes))
	#ma=200000
	if desDict=={} or desDict1=={}:
		return False
	else:
		#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		#matches = bf.match(pdes,ddes)
		#matches = sorted(matches, key = lambda x:x.distance)
		#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
		#print()
		firstarray=np.zeros((len(tempTuple),128))
		i=0
		for p in tempTuple:
			po=desDict[(p[0],p[1])]
			j=0
			for p in po:
				firstarray[i][j]=p
				j=j+1
			i=i+1
		
		
		#print(firstarray.shape)
		secondarray=np.zeros((len(tempTuple1),128))	
		i=0
		for q in tempTuple1:
			#print(q)
			la=desDict1[(q[0],q[1])]
			j=0
			for p in la:
				secondarray[i][j]=p
				j=j+1
			i=i+1
		
		pipp=0
		pipw=0
		min=1
		i=0
		j=0
		innerPointX=0
		innerPointY=0
		finalList=[]
		finalFirst=[]
		finalSecond=[]
		firstarray = firstarray / np.linalg.norm(firstarray)
		secondarray = secondarray / np.linalg.norm(secondarray)
		print(len(tempTuple),firstarray.shape,len(tempTuple1),secondarray.shape)
		for k in tempTuple:
			j=0
			for y in tempTuple1:
				tt=np.sum((firstarray[i]-secondarray[j])**2)
				if tt<min:
					min=tt
					innerPointX=y[0]
					innerPointY=y[1]
				j=j+1
				#print (min)
			finalList.append(min)
			finalSecond.append((k[0],k[1]))
			finalFirst.append((innerPointX,innerPointY))
			i=i+1
				
	
	vis = np.concatenate((secondImg1, color_img2), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();	
	
def getdescriptor(imarr,i,j):
	#print("Hello")
	vec = [0]*16
	vec[0] = localdir(i-8,j-8,imarr)
	vec[1] = localdir(i-8,j-4,imarr)
	vec[2] = localdir(i-4,j-8,imarr)
	vec[3] = localdir(i-4,j-4,imarr)
	vec[4] = localdir(i-8,j,imarr)
	vec[5] = localdir(i-8,j+4,imarr)
	vec[6] = localdir(i-4,j,imarr)
	vec[7] = localdir(i-4,j+4,imarr)
	vec[8] = localdir(i,j-8,imarr)
	vec[9] = localdir(i,j-4,imarr)
	vec[10] = localdir(i,j,imarr)
	vec[11] = localdir(i,j+4,imarr)
	vec[12] = localdir(i+4,j-8,imarr)
	vec[13] = localdir(i+4,j-4,imarr)
	vec[14] = localdir(i+4,j,imarr)
	vec[15] = localdir(i+4,j+4,imarr)
	
	return [val for subl in vec for val in subl]

	
def direction(i,j,imarr):
		#print(imarr.shape)
		mij=0
		theta=0
		#if i<470 and j<630:
		mij = math.sqrt((imarr[i+1,j]-imarr[i-1,j])**2 
				+(imarr[i,j+1]-imarr[i,j-1])**2)
		theta = math.atan((imarr[i,j+1]-imarr[i,j-1])
				/(imarr[i+1,j]-imarr[i-1,j]))

		return mij,theta

	
	
def localdir(i,j,imarr):
	#print(i,j)
	P = math.pi
	localDir = [0]*8

	for b in range(i,i+4):
		for c in range(j,j+4):
			m,t = direction(b,c,imarr)
			t=math.degrees(t)
			#print(t)
			if t>=0 and t<=45:
				localDir[0]+=m
			elif t>45 and t<=90:
				localDir[1]+=m
			elif t>90 and t<=135:
				localDir[2]+=m
			elif t>135 and t<180:
				localDir[3]+=m
			elif t>-180 and t<=-135: 
				localDir[4]+=m
			elif t>-135 and t<=-90:
				localDir[5]+=m
			elif t>-90 and t<=-45:
				localDir[6]+=m	
			elif t>-45 and t<=0:
				localDir[7]+=m
	return localDir

	
if __name__=="__main__":
	imgraw=input("imgage Input\n")
	harrisCornerDetection(cv2.imread(imgraw))
	
	