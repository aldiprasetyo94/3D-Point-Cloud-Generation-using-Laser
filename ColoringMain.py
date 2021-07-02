import cv2 as cv2
import numpy as np
import time


print('\n--- National Taiwan University of Science and Technology ---')
print('Name: 林超良 \t\t\t Department: Mechanical Engineering')
print('Student ID: M10803001 \t Assignment: Final Project - Reconstruct 3D from stereoscopic images and colorize the result')
print('---------------------------------------Coloring Point Cloud---------------------------------------------------------\n')
print('\nRequired Library:')
print('Opencv 3.4.2')
print('Numpy 1.18.1\n')


ColoredRefImg_FileName = 'Texture.JPG'
XYZFileName = 'pcd.xyz'
ColoredPcd_FileName = 'pcdcolor'

print('Reference Image File for Coloring\t:', ColoredPcd_FileName)
print('Point Cloud XYZ File\t\t\t\t:', XYZFileName)
print('Point Cloud Output file (with color):', ColoredPcd_FileName, '.xyz')


#1.Initial 6 projective points
print('\nStep 1. Initialing 6 projective points')
#3DPoints(Pcd)
X1 = [4.351741, -38.741188, 176.25827, 1] #tangan kanan ok
X2 = [14.728344, -38.707371, 176.216568, 1] #antara bunga dan pipi
X3 = [-7.161305, 51.282734, 166.170502, 1] #antara kaki kiri dan tangan ok
X4 = [10.390069, 52.732937, 172.349869, 1] #nose ok
X5 = [29.56159, 50.122131, 171.518616, 1] #pipi kiri ok
X6 = [-19.72913, 8.650562, 165.406662, 1] #pip kanan ok
X7 = [32.790901, 35.704632, 172.789932, 1]# bahu kanan ok


#2DPoints(Pcd)
uv1 = [1965, 1453, 1] #antara kaki kiri dan tangan ok
uv2 = [2189, 1453, 1] #nose ok
uv3 = [1665, 3363, 1]#pipi kiri ok
uv4 = [2047, 3359, 1] #pipi kanan ok
uv5 = [2485, 3365, 1] #bahu kanan ok
uv6 = [1397, 2419, 1] #tangan kanan ok
uv7 = [2549, 3067, 1] #antara bunga dan pipi



print('\nStep 2. Calculating PUV matrix')
#PUV Matrix
zero = [0,0,0,0]
def uX(uv,noOfCol,X):
    result=[]
    for n in range(0,4):
        result.append(-1 * uv[noOfCol] * X[n])
    return result

Puvmatrix = [X1 + zero + uX(uv1, 0, X1),
             zero + X1 + uX(uv1, 1, X1),
             X2 + zero + uX(uv2, 0, X2),
             zero + X2 + uX(uv2, 1, X2),
             X3 + zero + uX(uv3, 0, X3),
             zero + X3 + uX(uv3, 1, X3),
             X4 + zero + uX(uv4, 0, X4),
             zero + X4 + uX(uv4, 1, X4),
             X5 + zero + uX(uv5, 0, X5),
             zero + X5 + uX(uv5, 1, X5),
             X6 + zero + uX(uv6, 0, X6),
             zero + X6 + uX(uv6, 1, X6),
             X7 + zero + uX(uv7, 0, X7),
             zero + X7 + uX(uv7, 1, X7)
             ]

print('\nStep 3. Calculating Projective matrix P')
def Projection(PUVmatrix):
    U, s, VH = np.linalg.svd(PUVmatrix, full_matrices=True)
    V = VH.T.conj()

    P = np.array([[V[0][11], V[1][11], V[2][11], V[3][11]],
         [V[4][11], V[5][11], V[6][11], V[7][11]],
         [V[8][11], V[9][11], V[10][11], V[11][11]]
        ])
    P = P / P[2][3]

    return P

P =Projection(Puvmatrix)
print('Projective Matrix:')
print(P)


print('\nStep 4. Reading XYZ File.')
Xworld = np.loadtxt(XYZFileName, comments="#", delimiter=" ", unpack=False)
B=[]
for pointCloudNo in range(len(Xworld)):
    B.append(np.append([Xworld[pointCloudNo]], [1]))    #change every row of content in the XYZ file become (x y z 1), because in the formula we need 1x4 matrix of World coordinate
B = np.array(B)

print('\nStep 5. Projecting [Ximg = P.Xworld]')
Ximglist =[]
for NoOfPointCloud in range(len(B)):
    Ximg = np.dot(P, np.transpose(B[NoOfPointCloud])) #calculating Ximg
    Ximg= Ximg/Ximg[2] #normalize

    Ximg[0] = round(Ximg[0])    #Rounding the pixel, because pixel has to be non decimal number
    Ximg[1] = round(Ximg[1])
    Ximg[2] = round(Ximg[2])

    Ximglist.append([Ximg[0], Ximg[1], Ximg[2]]) #Saving all of projecting X Y pixel

Ximglist = np.array(Ximglist) #(numpy the list)


print('\nStep 6. Reading Colored Reference Image.')
Texture = cv2.imread(ColoredRefImg_FileName)
img_height = Texture.shape[0]
img_width = Texture.shape[1]
print('Image Height:', img_height)
print('Image width:', img_width)


print('\nStep 7. Coloring.')
start_time = time.time() #For checking the duration of coloring (Start Time)

PCD_color = [] #For saving the PCD Color information
process = 0
for pointCloudNo in range(len(Xworld)):

    PCD_R = [] #just temporary list for saving the PCD_Color
    PCD_RG=[]
    PCD_RGB = []

    if(0 <= (int(Ximglist[pointCloudNo][1]))<img_height and 0 <= (int(Ximglist[pointCloudNo][0]))< img_width):
        PCD_R = np.append(Xworld[pointCloudNo],[Texture[int(Ximglist[pointCloudNo][1]), int(Ximglist[pointCloudNo][0]), 2]])    #Red
        PCD_RG = np.append(PCD_R, [Texture[int(Ximglist[pointCloudNo][1]), int(Ximglist[pointCloudNo][0]), 1]])                 #Red Green
        PCD_RGB = np.append(PCD_RG, [Texture[int(Ximglist[pointCloudNo][1]), int(Ximglist[pointCloudNo][0]), 0]])               #Red Green Blue

    else: #if the projective pixel is out of range of reference image size, then it will be colored as black
        PCD_R = np.append(Xworld[pointCloudNo], 0)
        PCD_RG = np.append(PCD_R, 0)
        PCD_RGB = np.append(PCD_RG, 0)
    #Storing the list of PCD RGB
    PCD_color.append(PCD_RGB)

    #For Showing the progress of coloring
    if pointCloudNo / (len(Xworld)) > 0.25 and process == 0:
        print(int((pointCloudNo/(len(Xworld)))*100), '% (', pointCloudNo, '/', len(Xworld), 'Pointclouds )')
        process = process + 1
    elif pointCloudNo / (len(Xworld)) > 0.5 and process == 1:
        print(int((pointCloudNo / (len(Xworld))) * 100), '% (', pointCloudNo, '/', len(Xworld), 'Pointclouds )')
        process = process + 1
    elif pointCloudNo / (len(Xworld)) > 0.75 and process == 2:
        print(int((pointCloudNo / (len(Xworld))) * 100), '% (', pointCloudNo, '/', len(Xworld), 'Pointclouds )')
        process = process + 1
    elif pointCloudNo / (len(Xworld)) > 0.9 and process == 3:
        print(int((pointCloudNo / (len(Xworld))) * 100), '% (', pointCloudNo, '/', len(Xworld), 'Pointclouds )')
        process = process + 1
    elif pointCloudNo / (len(Xworld)-1) == 1:
        print('Finished the coloring.')

end_time = time.time()
print('Coloring Point Clouds duration:', int((end_time-start_time)/60), 'minutes', int((end_time-start_time) % 60), 'seconds')

print('\nStep 8. Saving the PCD (XYZRGB)')
np.savetxt('{}.xyz'.format(ColoredPcd_FileName), PCD_color, delimiter=' ')
print('[{}.xyz] file has been saved.'.format(ColoredPcd_FileName))