import numpy as np
import cv2
import math
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
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
	Ixx=cv2.GaussianBlur(Ixx,(3,3),0)
	Iyy=cv2.GaussianBlur(Iyy,(3,3),0)
	Ixy=cv2.GaussianBlur(Ixy,(3,3),0)
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
def getKeypoints(img,height,width,R,thres,maxim):
	tempTuple=[]
	rows,cols=R.shape
	for y in range(0,rows):
		for x in range(0,cols):
			if(R[y,x]>0):
				tempTuple.append((x,y))
	return tempTuple
	
def visualizing(img,tupletemp):
	for i in tupletemp:
		y=i[1]
		x=i[0]
		img.itemset((y,x,0),0)
		img.itemset((y,x,1),0)
		img.itemset((y,x,2),255)
		cv2.rectangle(img,(x-2,y-2),(x+2,y+2),255,1)
	return img

'''
Adaptive Maximum Supression
'''
def AdapativeMeasure(harris_response, corner_points, n):
    R = []
    responses = harris_response[corner_points[0], corner_points[1]]
    corner_points = np.hstack((corner_points[0][:, None], corner_points[1][:, None]))
    
    for (i, (y, x)) in enumerate(corner_points):
        bigger_neighbors = corner_points[responses > responses[i]]
        
        if bigger_neighbors.shape[0] == 0:
            radius = np.inf
        else:
            radius = np.sum((bigger_neighbors - np.array([y, x]))**2, 1)
            radius = radius.min()
        R.append(radius)
    
    n = min(len(R), n)
    p = np.argpartition(-np.asarray(R), n)[:n]
	
    z = corner_points[p]
    out_c = []
	
    for el in z:
       x = [el[1], el[0]]
       out_c.append(x)

    return out_c
	
	
	
'''
Converting the Dictionary to NumpyArray
Input:Key points for the image and Dictionary
Output:Numpy Array of descriptors
'''	
def convertingDictToArray(tempTuple,desDicto):
	#temp=np.zeros((len(tempTuple),128))
	holymollyConvert=[]
	i=0
	desDict = desDicto
	for p in tempTuple:
		po=desDict[(p[0],p[1])]
		holymollyConvert.append(po)
	return np.array(holymollyConvert)

'''
Calculating the SSDMeasure ,Ratio Test for each Key Point
Input:First Descriptors,Second Descriptors,first Keypoints,Second Keypoints
Output:finalList of minimum values,first Image key points,Second Image Keypoints
'''	
def SSDMeasure(firstarray,secondarray,temptuple,temptuple1):
	i=0
	threshold=0.7
	ratioTest=[]
	dmatch=[]
	tempIndex=0
	diff=[]
	minvalue=0
	for k in firstarray:
		diff=[]
		minvalue=0
		for y in secondarray:
			tt=np.sum((k-y)**2)
			diff.append(tt)
			
		diff=sorted(diff)
		for op in diff:
			if op>threshold:
				minvalue=op
				break
		dmatch.append(cv2.DMatch(i,diff.index(minvalue),minvalue))
		ratioTest.append(diff[0]/diff[1])
		i=i+1
	return ratioTest,dmatch
	
	
'''
Getting the Second Input Image Detecting Harris Corner 
Input:Image
Output:Result
'''	
def harrisCornerDetection(imgo):
	imgraw=input("Second Input\n")
	secondImg1=cv2.imread(imgraw)
	
	## Getitng Gray Scale images  for Images
	img,height,width=getGrayImage(imgo)
	secondImg,secondHeight,secondwidth=getGrayImage(secondImg1)
	
	##Getting Gradients
	dy,dx,Ixx,Iyy,Ixy=gradient_gaussian(img)
	dy2,dx2,Ixx2,Iyy2,Ixy2=gradient_gaussian(secondImg)
	
	##Threoshold
	thres=0.4
	##Drawing Keypoints on two Images
	color_img=imgo.copy()
	color_img2=secondImg1.copy()
	
	
	#Response Matrix
	R=response_matrixx(Ixx,Iyy,Ixy)
	R1=response_matrixx(Ixx2,Iyy2,Ixy2)
	##Getting Maximum Value of R
	firstImgMax=maximumRValue(R)
	secondImgMax=maximumRValue(R1)
	tempR=R>int(thres*firstImgMax)
	tempR1=R1>int(thres*secondImgMax)
	R=R*tempR
	R1=R1*tempR1
	adaptive=0
	if adaptive==1:
		anmsinput1=np.where(R>thres*firstImgMax)
		anmsinput2=np.where(R1>thres*secondImgMax)
		tempTuple2=AdapativeMeasure(R,anmsinput1,100)
		tempTuple3=AdapativeMeasure(R1,anmsinput2,100)
		color_img=visualizing(color_img,tempTuple2)
		color_img2=visualizing(color_img2,tempTuple3)
	else:
		#Non Maximum Supression 
		R=maxSupression(R)
		R1=maxSupression(R1)
		tempTuple2=getKeypoints(img,height,width,R,thres,firstImgMax)
		tempTuple3=getKeypoints(secondImg,secondHeight,secondwidth,R1,thres,secondImgMax)
		color_img=visualizing(color_img,tempTuple2)
		color_img2=visualizing(color_img2,tempTuple3)
	
	
	##Displaying Images with keypoints
	
	vis = np.concatenate((imgo, color_img), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	
	
	vis = np.concatenate((secondImg1, color_img2), axis=1)
	cv2.imshow("Test",vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	row,col=img.shape
	tempTuple=[]
	tempTuple1=[]
	
	for i in tempTuple2:
		if((i[0]+10>row) or (i[0]-10<0) or (i[1]+10>col) or (i[1]-10<0)):
			continue
		#print(i[0],i[1])
		tempTuple.append((i[0],i[1]))
	
	for i in tempTuple3:
		if((i[0]+10>row) or (i[0]-10<0) or (i[1]+10>col) or (i[1]-10<0)):
			continue
		tempTuple1.append((i[0],i[1]))
		
	
	
	#Getting Descriptors
	desDict = {}
	desDict1 = {}
	print("Calucalting Descriptors for First Image")
	for q in tempTuple:
		#print("desc",q[0],q[1])
		desDict[(q[0],q[1])]=getdescriptor(img,q[0],q[1])
	print("Number of Descriptors in first Image",len(desDict))
	
	print("Calucalting Descriptors for Second Image")
	for e in tempTuple1:
		desDict1[(e[0],e[1])]=getdescriptor(secondImg,e[0],e[1])
	print("Number of Descriptors in Second Image",len(desDict1))
	
	#calculating the matches
	finalList=[]
	finalfirst=[]
	finalsecond=[]
	ratioTest=[]
	matches=[]
	if desDict=={} or desDict1=={}:
		return False
	else:
		firstarray=convertingDictToArray(tempTuple,desDict) 
		secondarray=convertingDictToArray(tempTuple1,desDict1)
		firstarray=np.clip(firstarray,0,0.2)
		secondarray=np.clip(secondarray,0,0.2)
		#firstarray = firstarray / np.linalg.norm(firstarray)
		#secondarray = secondarray / np.linalg.norm(secondarray)
		ratioTest,matches=SSDMeasure(firstarray,secondarray,tempTuple,tempTuple1)
	
	img=img.astype(np.uint8)
	secondImg=secondImg.astype(np.uint8)
	#Making Keypoint list using CV2.keypoint
	KeypointsImg1=[]
	KeypointsImg2=[]
	for keyfir in tempTuple:
		KeypointsImg1.append(cv2.KeyPoint(keyfir[0],keyfir[1],1))
	
	
	for keysec in tempTuple1:
		KeypointsImg2.append(cv2.KeyPoint(keysec[0],keysec[1],1))
		
	matches = sorted(matches, key = lambda x:x.distance)
	outimage = cv2.drawMatches(img,KeypointsImg1,secondImg,KeypointsImg2,matches[:10],None, flags=2)
	#plt.imshow((outimage).astype(np.uint8)),plt.show()
	cv2.imshow("Test",outimage)
	cv2.waitKey(0)
	cv2.destroyAllWindows();
	
	
	
	#cv2.drawMatches(img,matchpointsImg1,secondImg,matchpointsImg2,matches,FinalImage)
	
			
'''
Gettting the Descriptors for the Following keypoints
Input:Image,x,y
Output:Descriptor
'''	
def getdescriptor(imarr,i,j):
	#print("Hello")
	vec = [0]*16
	vec[0] = binsContainer(i-8,j-8,imarr)
	vec[1] = binsContainer(i-8,j-4,imarr)
	vec[2] = binsContainer(i-4,j-8,imarr)
	vec[3] = binsContainer(i-4,j-4,imarr)
	vec[4] = binsContainer(i-8,j,imarr)
	vec[5] = binsContainer(i-8,j+4,imarr)
	vec[6] = binsContainer(i-4,j,imarr)
	vec[7] = binsContainer(i-4,j+4,imarr)
	vec[8] = binsContainer(i,j-8,imarr)
	vec[9] = binsContainer(i,j-4,imarr)
	vec[10] = binsContainer(i,j,imarr)
	vec[11] = binsContainer(i,j+4,imarr)
	vec[12] = binsContainer(i+4,j-8,imarr)
	vec[13] = binsContainer(i+4,j-4,imarr)
	vec[14] = binsContainer(i+4,j,imarr)
	vec[15] = binsContainer(i+4,j+4,imarr)
	
	imx=[val for subl in vec for val in subl]
	imx=np.array(imx)
	imx=imx / np.linalg.norm(imx)
	return imx
	#return [val for subl in vec for val in subl]

'''
Calculating Magnitutde and direction 
Input:x,y,Img
Output:Magnitude,Direction
'''	
def direction(i,j,imarr):
		#print("Gradient",i,j)
		mij=0
		theta=0
		mij = np.sqrt((imarr[i+1,j]-imarr[i-1,j])**2 
				+(imarr[i,j+1]-imarr[i,j-1])**2)
		theta = ((np.arctan2((imarr[i,j+1]-imarr[i,j-1])
				,(imarr[i+1,j]-imarr[i-1,j])))*180/np.pi)%360

		return mij,theta

	
'''
Getitng the patch around the Keypoints and creating Bins
Input:x,y,Img
Output:Bins for the particular Cell
'''	
def binsContainer(i,j,imarr):
	#print(i,j)
	bins = [0]*8
	for b in range(i,i+4):
		for c in range(j,j+4):
			m,t = direction(b,c,imarr)
			#t=math.degrees(t)
			#print(t)
			if t>=0 and t<=45:
				bins[0]+=m
			elif t>45 and t<=90:
				bins[1]+=m
			elif t>90 and t<=135:
				bins[2]+=m
			elif t>135 and t<=180:
				bins[3]+=m
			elif t>180 and t<=225: 
				bins[4]+=m
			elif t>225 and t<=270:
				bins[5]+=m
			elif t>270 and t<=315:
				bins[6]+=m	
			elif t>315 and t<=360:
				bins[7]+=m
	return bins

	
'''
Main Function Takes the First input Image
'''	
if __name__=="__main__":
	imgraw=input("imgage Input\n")
	harrisCornerDetection(cv2.imread(imgraw))
	
	