
import numpy as np
import cv2

import errno
import os
import sys


from glob import glob

def readImages(image_dir):
    """This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]
    #images_names = []
    #images_names = [ [images_names,f]
              #for f in image_files]
    #print image_files
    

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))

    return images, image_files


def faceDetection(inputPath, entirePhoto_outputPath, faceCrops_outputPath):
    face_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_eye.xml')
    images,images_names = readImages(inputPath)
    entire_image_output_dir = entirePhoto_outputPath
    cropFaces_images_output_dir = faceCrops_outputPath
    #print images[0].shape,images[1].shape
    #'''
    for varI in range(len(images)):
        img = images[varI]
        img_name = os.path.basename(images_names[varI])
        img_name_no_ext, file_extension = os.path.splitext(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_copy = img.copy()
        faces = face_cascade.detectMultiScale(gray, 1.3, 10) #1.01, 40)

        increase_ratio_size_of_face_crop_x = 3
        increase_ratio_size_of_face_crop_y = 4
        #print images_names
        print images_names[varI], img_name, img_name_no_ext
        print faces
        varJ=0
        for (x,y,w,h) in faces:
            w_1 = round(w*increase_ratio_size_of_face_crop_x)
            h_1 = round(h*increase_ratio_size_of_face_crop_y)
            x_1 =  max(0, round(x - (w_1-w)/2.0) )
            y_1 =  max(0, round(y - (h_1-h)/2.0) )
            x_2 = min(gray.shape[1] ,x_1+w_1)
            y_2 = min(gray.shape[0] ,y_1+h_1)
            w_1 = x_2-x_1
            h_1 = y_2-y_1
            
            x_1 = int(x_1)
            y_1 = int(y_1)
            w_1 = int(w_1)
            h_1 = int(h_1)
            x_2 = int(x_2)
            y_2 = int(y_2)
            
            #print x_1,y_1,w_1,h_1
            
            roi_gray = gray[y_1:y_1+h_1, x_1:x_1+w_1]
            roi_color = img[y_1:y_1+h_1, x_1:x_1+w_1]
            cv2.imwrite(cropFaces_images_output_dir+'/'+img_name_no_ext+'_'+str(varJ)+'.jpg',roi_color)
            
            #draw rectangle
            cv2.rectangle(img_copy,(x_1,y_1),(x_1+w_1,y_1+h_1),(0,255,0),10)
            
            #eye detection:
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            varJ = varJ+1
        print entire_image_output_dir+'/'+img_name
        cv2.imwrite(entire_image_output_dir+'/'+img_name,img_copy)
    
if __name__ == "__main__":
    
    inputPath = './input'
    entirePhoto_outputPath = './output/entire_photo'
    faceCrops_outputPath = './output/face_crops'
    faceDetection(inputPath, entirePhoto_outputPath, faceCrops_outputPath)
    
    #face_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_eye.xml')
    #images,images_names = readImages('./input')
    #entire_image_output_dir = './output/entire_photo'
    #cropFaces_images_output_dir = './output/face_crops'
    ##print images[0].shape,images[1].shape
    ##'''
    #for varI in range(len(images)):
        #img = images[varI]
        #img_name = os.path.basename(images_names[varI])
        #img_name_no_ext, file_extension = os.path.splitext(img_name)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_copy = img.copy()
        #faces = face_cascade.detectMultiScale(gray, 1.3, 10) #1.01, 40)

        #increase_ratio_size_of_face_crop_x = 3
        #increase_ratio_size_of_face_crop_y = 4
        ##print images_names
        #print images_names[varI], img_name, img_name_no_ext
        #print faces
        #varJ=0
        #for (x,y,w,h) in faces:
            #w_1 = round(w*increase_ratio_size_of_face_crop_x)
            #h_1 = round(h*increase_ratio_size_of_face_crop_y)
            #x_1 =  max(0, round(x - (w_1-w)/2.0) )
            #y_1 =  max(0, round(y - (h_1-h)/2.0) )
            #x_2 = min(gray.shape[1] ,x_1+w_1)
            #y_2 = min(gray.shape[0] ,y_1+h_1)
            #w_1 = x_2-x_1
            #h_1 = y_2-y_1
            
            #x_1 = int(x_1)
            #y_1 = int(y_1)
            #w_1 = int(w_1)
            #h_1 = int(h_1)
            #x_2 = int(x_2)
            #y_2 = int(y_2)
            
            ##print x_1,y_1,w_1,h_1
            
            #roi_gray = gray[y_1:y_1+h_1, x_1:x_1+w_1]
            #roi_color = img[y_1:y_1+h_1, x_1:x_1+w_1]
            #cv2.imwrite(cropFaces_images_output_dir+'/'+img_name_no_ext+'_'+str(varJ)+'.jpg',roi_color)
            
            ##draw rectangle
            #cv2.rectangle(img_copy,(x_1,y_1),(x_1+w_1,y_1+h_1),(0,255,0),10)
            
            ##eye detection:
            ##eyes = eye_cascade.detectMultiScale(roi_gray)
            ##for (ex,ey,ew,eh) in eyes:
                ##cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #varJ = varJ+1
        #cv2.imwrite(entire_image_output_dir+'/'+img_name,img_copy)
        ##cv2.imwrite(os.path.join('./output/entire_photo', '{}_diff1.png'.format(video_dir)), diff1)

        ##cv2.imshow('img',img)
        ##cv2.waitKey(0)
        ##cv2.destroyAllWindows()
        ##'''