import cv2 as cv2
import numpy as np
import os
import time


print('\n--- National Taiwan University of Science and Technology ---')
print('Name: Lin-Chaoliang \t\t\t Department: Mechanical Engineering')
print('Student ID: M10803001 \t Assignment: Final Project - Reconstruct 3D from stereoscopic images and colorize the result')
print('-----------------------------------------------Generating Point Cloud---------------------------------------------\n')

print('\nRequired Library:')
print('Opencv 3.4.2')
print('Numpy 1.18.1\n')




def ImgFileName(no_of_file):
    a = str(no_of_file)
    number_length = len(a)
    zero_number = "0" * (3 - (number_length))
    final_number = zero_number + a
    return final_number

def GroundTruth (u,v,P,up,vp,Pp):

    p1 = np.transpose(P[0])
    p2 = np.transpose(P[1])
    p3 = np.transpose(P[2])


    pp1 = np.transpose(Pp[0])
    pp2 = np.transpose(Pp[1])
    pp3 = np.transpose(Pp[2])



    A = np.array([(u*p3) - p1,
                  (v*p3)-p2,
                  (up*pp3) -pp1,
                  (vp*pp3) -pp2])

    U,S,Vh = np.linalg.svd(A)
    V = Vh.T.conj()
    Vn = V/V[3][3]
    X = [Vn[0][3],Vn[1][3],Vn[2][3]]

    return X




start_time = time.time()

MatrixFileName = 'b.txt'
print('\nMatrix Parameter File\t\t:',MatrixFileName)
print('Camera Left Image Folder\t: L')
print('Camera Right Image Folder\t: R\n')

#Left camera
L_intrinsic = np.loadtxt(MatrixFileName, comments="#", delimiter=" ", unpack=False, skiprows=2, max_rows=3)
L_extrinsic = np.loadtxt(MatrixFileName, comments="#", delimiter=" ", unpack=False, skiprows=12, max_rows=3)


#Right Camera
R_intrinsic = np.loadtxt(MatrixFileName, comments="#", delimiter=" ", unpack=False, skiprows=6, max_rows=3)
R_extrinsic = np.loadtxt(MatrixFileName, comments="#", delimiter=" ", unpack=False, skiprows=16, max_rows=3)

#F Matrix
F_matrix = np.loadtxt(MatrixFileName, comments="#", delimiter=" ", unpack=False, skiprows=22, max_rows=3)

#P each camera P = K[R|t]
P = np.dot(L_intrinsic,L_extrinsic)


Pp = np.dot(R_intrinsic,R_extrinsic)



Left_no_files = len(os.listdir('L'))
Right_no_files = len(os.listdir('R'))


print('Generating Point Cloud:')
pcd = [] #initial for point clouds
for Img_no in range(Left_no_files): #Read all files one by one  Left_no_files

    print('No of image:', Img_no, '/', Left_no_files)


    Img_file_name = ImgFileName(Img_no)
    L_img = cv2.cvtColor(cv2.imread('L/L'+Img_file_name+'.JPG'), cv2.COLOR_BGR2GRAY)
    R_img = cv2.cvtColor(cv2.imread('R/R' + Img_file_name + '.JPG'), cv2.COLOR_BGR2GRAY)


    L_img_laser = []
    R_img_laser = []
    for y in range(len(L_img)):
        for x in range(len(L_img[y])):
            if L_img[y][x]>100:
                L_img_laser.append([x,y,1])
                break

    for yp in range(len(R_img)):
        for xp in range(len(R_img[yp])):
            if R_img[yp][xp]>100:
                R_img_laser.append([xp,yp,1])
                break

    for n in range (len(L_img_laser)):
        for m in range(len(R_img_laser)):
            if -0.1<(np.dot(np.dot(np.transpose(R_img_laser[m]),F_matrix),L_img_laser[n]))<0.1:
                pcd.append(GroundTruth(u=L_img_laser[n][0], v=L_img_laser[n][1], P=P, up=R_img_laser[m][0], vp=R_img_laser[m][1], Pp=Pp))
                break




np.savetxt('{}.xyz'.format('pcd'), pcd, delimiter=' ')
print('\n{}.xyz has been saved', format('pcd'))
end_time = time.time()
print('Generated Point Clouds duration:',int(end_time-start_time)/60,'minutes')
print('\n Program is closed in 10 seconds...')
time.sleep(10)













