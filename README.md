# Sift_Descriptors_Harris_Corner

Two Input Images are taken when Executing the program

In first Part Calculating the Harris Corner Points:
From main Method these harrisCornerDetection() function is called.
And most of the calls to other functions are done through this.


Step 1:
Images are taken as input and converted to GrayScale and Converted to float32.

Step 2:

Firstly I calculated the the Image Gradient for the image
dx and dy 
in the gradient_gaussian() function .
After which Ixx ,Ixy and Iyy is also calculated for tthe images in the same function.

Then Applied GaussianBlur on the Ixx,Iyy,Ixy to get the Gaussian matrix.

Step 3: 
Now for every Ixx,Iyy,Ixy Calculated  the determinant ,Trace and Response function in response_matrixx()

Step4:

The Reponse Matrix that we get from the earlier step is then maximum Supressed In which only those elements are taken into account whose value is maximum in surrounding 3*3 matrix .This is done in maxSupression() function which returns us the the Response matrix with supressed values.
These are done on the two images .

Step 5:
All the Reponse function which we get during the step 4 , maximum value is calculated in that matrix in maximumRValue() function.

step 6:
Then getting the keypoints that is greater than the (threshold*maximum Value in the R matrix) and storing all those points in one list with tuple
And Drawing all the points accordingly on the image.
Here you get two pop up first one is first image and then the next Image with all the keypoints displayed.

Step 7:
In these After getting all the keypoints I needed to remove those points that are nearer to the border as we might go out of the image while calculating the descriptors and all other equation.

Step 8:
After filtering out all the border points we need to get the descriptors for the keypoints that is being stored with us.
this is done by calling getdescriptor() function. 

These getdescriptor() function takes keypoints one by one and then creates a 16*16 window and then this 16*16 is dividied into 16 4*4 cells.

For each cell :
binsContainer() function is called. Here we create 8 bins ranging from 0 to 360 degree.In this function the it loops through the cell and then it calls direction() function for each point in that cell and calculates maginitude and theata.
Depending upon the value of theata the magnitude is added to the appropriate bins.
This function returns list with the bin for that point .
binsContainer() is called for all the cells that are in the patch and hence we get all those bins for each cell.
After getting these we need to combine into one list and we get 128 descriptors and normalise after which we return this value.

Step 8 is done for all the keypoints.
We get dictionary containing the keypoints and associated descriptors which is a list.

Step 9:
In this we converted all the list that belongs to corrresponding keypoints to numpy.We clip the histogram to all 0.2 .

Step 10:
Here we calculate the SSD(sum of Square Difference) by calling SSDMeasure() function .
Here we get substract the discriptors from one point in firstimage to all points in second image to get the minimum distance.
After getting the minimum distance we apply a threshold .Only those points are matched whose minimum value is greater. 
We also calculate the ratio test. By dividing the best Matxh/Second Best match for each key point.
Here we also create a list containing Dmatch object that accepts the index of first image ,index of second image,mimimumvalue.

Step 11:
We got all the matches in the earlier Step.
To use drawmatches we need to get list containing keyPoint objects.
We create a two list for two images that will contain corresponding Keypoints .
Keypoints take the arguments  firstImage, first image Keypoints object list, Second Image,  Second image Keypoints object list, Matches ,output image, flags.
This is the last step.
The output will show the two images with mapping from one image to another.

-------------------------------------------------End------------------------------------------------------

