import numpy as np
import cv2
import math
from scipy.ndimage import maximum_filter
np.seterr(divide='ignore', invalid='ignore')


'''
Getting the GrayScale Image After Reading \
Input:Image
Output:Gray Image,Height,
'''
def getGrayImage(img):
	img=np.float32(img)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	height=img.shape[0]
	width=img.shape[1]
	return img,height,width
	
'''
Calculating the Gradient and GaussianBlur
Input:GrayScale
Output:GradientX,GradientY,SquareX,SquareY,XY
'''
def gradient_gaussian(img):
	dy,dx=np.gradient(img)
	Ixx=dx**2;
	Iyy=dy**2
	Ixy=dx*dy
	Ixx=cv2.GaussianBlur(Ixx,(3,3),2)
	Iyy=cv2.GaussianBlur(Iyy,(3,3),2)
	Ixy=cv2.GaussianBlur(Ixy,(3,3),2)
	return dy,dx,Ixx,Iyy,Ixy
	
'''
Getting the harris response Matrix
Input:SquareX,Squarey,Xy
output:Response matrix
'''
def response_matrixx(Ixx,Iyy,Ixy):
	det= (Ixx*Iyy)-(Ixy**2)
	trace= Ixx+Iyy
	R=det/trace
	return R

'''
Getting the maximum value in the Response Matrix
input:Response Matrix
output: Maximum Value
'''	
def maximumRValue(R):
	map(max, R)
	list(map(max, R))
	maxim=max(map(max, R))
	return maxim

'''
Max Supression Done on the Response Matrix
Input:Response Matrix
Output:Response Matrix with Max Supression
'''	
def maxSupression(R):
	localmax=maximum_filter(R,size=3)
	localFilter=R ==localmax
	R=R * localFilter
	return R

'''
Calculating the Keypoints for the Img
Input:Image,Height,Width,Response matrix,Threshold,Maximum Value,Color Image
Output:Keypoints in the Image and red dot is being placed within rectangle around keypoints
'''	
def getKeypoints(img,height,width,R,thres,maxim,color_img):
	tempTuple=[]
	for y in range(0,height,2):
		for x in range(0,width,2):
			if(R[y,x]>(maxim*thres)):
				if (y-8)<0 or (y+8)>img.shape[0] or (x-8)<0 or (x+8)>img.shape[1]:
					continue
				#finalimage[y,x]=r[y,x]
				tempTuple.append((y,x))
				color_img.itemset((y,x,0),0)
				color_img.itemset((y,x,1),0)
				color_img.itemset((y,x,2),255)
				cv2.rectangle(color_img,(x-2,y-2),(x+2,y+2),255,1)
	return tempTuple,color_img
	
	
'''
Converting the Dictionary to NumpyArray
Input:Key points for the image and Dictionary
Output:Numpy Array of descriptors
'''	
def convertingDictToArray(tempTuple,desDicto):
	temp=np.zeros((len(tempTuple),128))
	i=0
	desDict = desDicto
	for p in tempTuple:
		po=desDict[(p[0],p[1])]
		j=0
		for p in po:
			temp[i][j]=p
			j=j+1
		i=i+1
	return temp

'''
Calculating the SSDMeasure ,Ratio Test for each Key Point
Input:First Descriptors,Second Descriptors,first Keypoints,Second Keypoints
Output:finalList of minimum values,first Image key points,Second Image Keypoints
'''	
def SSDMeasure(firstarray,secondarray,tempTuple,tempTuple1):
	min=1
	secondMin=2
	i=0
	j=0
	innerPointX=0
	innerPointY=0
	finalList=[]
	finalFirst=[]
	finalSecond=[]
	ratioTest=[]
	#print(len(tempTuple),firstarray.shape,len(tempTuple1),secondarray.shape)
	for k in tempTuple:
		j=0
		for y in tempTuple1:
			tt=np.sum((firstarray[i]-secondarray[j])**2)
			if tt<min:
				secondMin=min
				min=tt
				innerPointX=y[0]
				innerPointY=y[1]
			j=j+1
			#print (min)
		ratioTest.append(min/secondMin)
		finalList.append(min)
		finalFirst.append((k[0],k[1]))
		finalSecond.append((innerPointX,innerPointY))
		i=i+1
	return finalList,finalFirst,finalSecond
	
'''
Getting the Second Input Image Detecting Harris Corner 
Input:Image
Output:Result
'''	
def harrisCornerDetection(imgo):
	imgraw=input("Second Input\n")
	secondImg1=cv2.imread(imgraw)
	img,height,width=getGrayImage(imgo)
	secondImg,secondHeight,secondwidth=getGrayImage(secondImg1)
	dy,dx,Ixx,Iyy,Ixy=gradient_gaussian(img)
	dy2,dx2,Ixx2,Iyy2,Ixy2=gradient_gaussian(secondImg)
	thres=0.2
	windowsize=3
	rmax=0
	color_img=imgo.copy()
	color_img2=secondImg1.copy()
	#hog_image=imgo.copy()
	R=response_matrixx(Ixx,Iyy,Ixy)
	R1=response_matrixx(Ixx2,Iyy2,Ixy2)
	firstImgMax=maximumRValue(R)
	secondImgMax=maximumRValue(R1)
	R=maxSupression(R)
	R1=maxSupression(R1)
	tempTuple,color_img=getKeypoints(img,height,width,R,thres,firstImgMax,color_img)
		
	vis = np.concatenate((imgo, color_img), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	
	tempTuple1,color_img2=getKeypoints(secondImg,secondHeight,secondwidth,R1,thres,secondImgMax,color_img2)
	vis = np.concatenate((secondImg1, color_img2), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	desDict = {}
	desDict1 = {}
	
	arr = img.astype(np.float64)
	arr2=secondImg.astype(np.float64)
	for q in tempTuple:		
		desDict[(q[0],q[1])]=getdescriptor(arr,q[0],q[1])
		
	for e in tempTuple1:
		desDict1[(e[0],e[1])]=getdescriptor(arr2,e[0],e[1])
		

	if desDict=={} or desDict1=={}:
		return False
	else:
		firstarray=convertingDictToArray(tempTuple,desDict) 
		secondarray=convertingDictToArray(tempTuple1,desDict1)
		firstarray = firstarray / np.linalg.norm(firstarray)
		secondarray = secondarray / np.linalg.norm(secondarray)
		firstarray=np.clip(firstarray,0,0.2)
		secondarray=np.clip(secondarray,0,0.2)
		finalList,finalfirst,finalsecond=SSDMeasure(firstarray,secondarray,tempTuple,tempTuple1)
			
'''
Gettting the Descriptors for the Following keypoints
Input:Image,x,y
Output:Descriptor
'''	
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

'''
Calculating Magnitutde and direction 
Input:x,y,Img
Output:Magnitude,Direction
'''	
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

	
'''
Getitng the patch around the Keypoints and creating Bins
Input:x,y,Img
Output:Bins for the particular Cell
'''	
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

'''
Main Function Takes the First input Image
'''	
if __name__=="__main__":
	imgraw=input("imgage Input\n")
	harrisCornerDetection(cv2.imread(imgraw))
	
	