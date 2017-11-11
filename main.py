import cv2
import numpy as np
import os
import errno
from os import path
import sys
from glob import glob
import faceDetectorCode
import math
import random
#from images2gif import writeGif
#import imageio

SRC_FOLDER = "input/sample"
FACE_DETECTION_ENTIRE_PHOTO_FOLDER = 'output/entire_photo'
SRC_FOLDER_CROP_FACES = "output/face_crops"
SRC_FOLDER_CROP_FACES_MASK = "output/segmented_output_mask"
OUT_FOLDER = "output/final_output"
OUT_GIF_FOLDER1 = "output/final_output/gif1"
OUT_GIF_FOLDER2 = "output/final_output/gif2"
OUT_MOTION_BLUR1_FOLDER = "output/final_output/motionBlurHor"
OUT_MOTION_BLUR2_FOLDER = "output/final_output/motionBlurVer"
OUT_MOTION_BLUR3_FOLDER = "output/final_output/motionBlurDiagLeft"
OUT_MOTION_BLUR4_FOLDER = "output/final_output/motionBlurDiagRight"
OUT_GAUSS_BLUR_FOLDER = "output/final_output/gaussBlur"
VIDEO_FOLDER1 = "cars"
VIDEO_FOLDER2 = "bokeh"

EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def modifyBackground(face_crop_image_files, face_crop_mask_image_files):
    
   backgroundEffect_Code = 0 
    
   print
   print
   for varI in  range(len(face_crop_image_files)):
       print face_crop_image_files[varI],face_crop_mask_image_files[varI]
       
       image_orig = cv2.imread(face_crop_image_files[varI])
       image = image_orig.copy()
       mask = cv2.imread(face_crop_mask_image_files[varI],0)#foreground is black, background is black
       ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
       inverted_mask = cv2.bitwise_not(mask)
       
       
       #camera motion blur operation:
       ksize = 701
       kernelVer = np.zeros((ksize,ksize),np.float32)
       kernelHor = np.zeros((ksize,ksize),np.float32)
       kernelDiagLeft = np.zeros((ksize,ksize),np.float32)
       kernelDiagRight = np.zeros((ksize,ksize),np.float32)
       
       for varJ in range(ksize):
           kernelHor[ksize/2,varJ] = 1/(ksize*1.0)
           kernelVer[varJ,ksize/2] = 1/(ksize*1.0)
           kernelDiagLeft[varJ,varJ] = 1/(ksize*1.0)
           kernelDiagRight[varJ,ksize-1-varJ] = 1/(ksize*1.0)
       
       image_inpaint = image_orig.copy()
       erode_kernel = np.ones((5,5),np.uint8)
       eroded_mask = cv2.erode(mask.copy(), erode_kernel,iterations = 5)
       width = mask.shape[1]
       height = mask.shape[0]
       for varK in range(mask.shape[0]):
        #for varJ in range(mask.shape[1]):
        rowArr = eroded_mask[varK,:]
        ranges = zero_runs(rowArr)
        #print ranges.shape
        #print ranges
        if(ranges.shape[0]>0):
            left = ranges[0][0]
            right = ranges[0][1]
            #if(varK==395):
                #print "a",left,right,width
            #if(right==width):
            right = right - 1
            width_of_zeros = right-left+1
            left_width = left
            right_width = width - right
            #print "xx",width,right,image_inpaint.shape
            pivot = math.floor(left_width + (left_width*width_of_zeros)/(left_width+right_width+0.000001))
            pivot = int(pivot)
            
            for varM in range(3): #rgb
                repeatArrNum = 1
                rowArr2 = image_inpaint[varK,:,varM]
                
                revArr_left = (rowArr2[0:left_width])
                ##revArr_left = revArr_left[::-1]
                #print "xx",rowArr2.shape,revArr_left.shape,left,left_width,width,pivot
                revArr_left = np.repeat(revArr_left, math.floor((pivot-left)/(left_width+0.00001) + repeatArrNum))
                #print left,pivot
                tmpW = image_inpaint[varK,left:pivot,varM].shape[0]
                #print "x",image_inpaint[varK,left:pivot,varM].shape,revArr_left.shape,tmpW  
                revArr_left = revArr_left[0:tmpW]
                #print left_width, pivot, right_width, varK,rowArr2.shape,varI
                #print image_inpaint[varK,left_width:pivot,0].shape
                #left_width=left_width
                #if(varK==395):
                    #print "x",image_inpaint[varK,left:pivot,varM].shape,revArr_left.shape,tmpW,left,pivot
                    #print tmpW, image_inpaint.shape
                image_inpaint[varK,left:pivot,varM] = revArr_left[0:tmpW]
                #print ranges,left,right,pivot
                
                revArr_right = (rowArr2[right:])
                #print "yy",rowArr2.shape,revArr_right.shape,right,right_width,width,pivot
                #revArr_right = revArr_right[::-1]
                revArr_right = np.repeat(revArr_right, math.floor((right-pivot)/(right_width+0.00001) + repeatArrNum))
                tmpW = image_inpaint[varK,pivot:right,varM].shape[0]
                #print image_inpaint[varK,pivot:right,varM].shape
                revArr_right = revArr_right[0:tmpW]
                #print revArr_right.shape
                #print right, pivot, right_width, varK,rowArr2.shape,varI
                #print image_inpaint[varK,left_width:pivot,0].shape
                #revArr_right = revArr_right[::-1]
                #if(varK==395):
                    #print "y",image_inpaint[varK,pivot:right,varM].shape,revArr_right.shape,tmpW,right,pivot
                    #print tmpW, image_inpaint.shape
                image_inpaint[varK,pivot:right,varM] = revArr_right
                #print ranges,left,right,pivot
            
            

        #if(m   ask[varK,varJ]==0):
            #start = 
      
      
       #print kernelVer
       #print kernelHor
       #print kernelDiagLeft
       #print kernelDiagRight
       
       #print varI
       
       
       translation_kernel = kernelHor
       image = image_orig.copy()
       motionBlur = cv2.filter2D(image_inpaint,-1,translation_kernel)
       foreground = cv2.bitwise_and(image, image, mask = inverted_mask)
       background = cv2.bitwise_and(motionBlur, motionBlur, mask = mask)
       out = cv2.bitwise_or(foreground, background)
       cv2.imwrite(path.join(OUT_MOTION_BLUR1_FOLDER,os.path.basename(face_crop_image_files[varI]) ), out )
       
       translation_kernel = kernelVer
       image = image_orig.copy()
       motionBlur = cv2.filter2D(image_inpaint,-1,translation_kernel)
       foreground = cv2.bitwise_and(image, image, mask = inverted_mask)
       background = cv2.bitwise_and(motionBlur, motionBlur, mask = mask)
       out = cv2.bitwise_or(foreground, background)
       cv2.imwrite(path.join(OUT_MOTION_BLUR2_FOLDER,os.path.basename(face_crop_image_files[varI]) ), out )
       
       translation_kernel = kernelDiagLeft 
       image = image_orig.copy()
       motionBlur = cv2.filter2D(image_inpaint,-1,translation_kernel)
       foreground = cv2.bitwise_and(image, image, mask = inverted_mask)
       background = cv2.bitwise_and(motionBlur, motionBlur, mask = mask)
       out = cv2.bitwise_or(foreground, background)
       cv2.imwrite(path.join(OUT_MOTION_BLUR3_FOLDER,os.path.basename(face_crop_image_files[varI]) ), out )
       
       
       translation_kernel = kernelDiagRight
       image = image_orig.copy()
       motionBlur = cv2.filter2D(image_inpaint,-1,translation_kernel)
       foreground = cv2.bitwise_and(image, image, mask = inverted_mask)
       background = cv2.bitwise_and(motionBlur, motionBlur, mask = mask)
       out = cv2.bitwise_or(foreground, background)
       cv2.imwrite(path.join(OUT_MOTION_BLUR4_FOLDER,os.path.basename(face_crop_image_files[varI]) ), out )
       
       
       
       
       
       #gaussian blur:
       #print ret
       #for varI in range(mask.shape[0]):
           #for varJ in range(mask.shape[1]):
               #if(mask[varI,varJ]!=0 and mask[varI,varJ]!=255):
                #print mask[varI,varJ]
       #print "x"
       #print mask
       #print "0"
       #print inverted_mask
       #print "y"
       
       #blurring operation
       sigma = 31
       ksize = 31
       blur = cv2.GaussianBlur(image_inpaint,ksize=(ksize,ksize),sigmaX = sigma, sigmaY = sigma, borderType = 0)
       #image = blur
       foreground = cv2.bitwise_and(image, image, mask = inverted_mask)
       background = cv2.bitwise_and(blur, blur, mask = mask)
       out = cv2.bitwise_or(foreground, background)
       cv2.imwrite(path.join(OUT_GAUSS_BLUR_FOLDER,os.path.basename(face_crop_image_files[varI]) ), out )
       
       
       #video texture based background:
       filename  = (os.path.splitext(os.path.basename(face_crop_image_files[varI]))[0])
       #print "x",filename
       videoTexture(image_orig.copy(), mask,filename, VIDEO_FOLDER1, OUT_GIF_FOLDER1  )
       
       filename  = (os.path.splitext(os.path.basename(face_crop_image_files[varI]))[0])
       #print "x",filename
       videoTexture(image_orig.copy(), mask,filename, VIDEO_FOLDER2, OUT_GIF_FOLDER2  )
  
def videoTexture(image_orig, mask,imgName, video_folder = "bokeh", output_folder = "output/final_output/gif1"):
    # read files from folder
    video_images = []
    folder = path.join("input/videos", video_folder)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            video_images.append(img)
    
    print len(video_images)

    # calculate bounding box of mask
    inverted_mask = cv2.bitwise_not(mask)
    B = np.argwhere(inverted_mask)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
    bb_inv_mask = inverted_mask[ystart:ystop, :]
    bb_mask = mask[ystart:ystop, :]
    bb_img_orig = image_orig[ystart:ystop, :]
    # bb_inv_mask = inverted_mask[ystart:ystop, xstart:xstop]
    # bb_mask = mask[ystart:ystop, xstart:xstop]
    # bb_img_orig = image_orig[ystart:ystop, xstart:xstop]

    # make both heights equal
    new_height = int(math.floor(bb_img_orig.shape[0] * 1.2))
    # video_images = cv2.pyrDown(video_images, dstsize = (new_height, new_height / video_images.shape[0] * video_images.shape[1]))

    # randomly position portrait on video image
    randInit = -1

    counter = 0
    output = []

    for img in video_images:
        # resize video images to new height
        img = cv2.resize(img, dsize=(new_height * img.shape[1]/ img.shape[0], new_height))
        if(randInit == -1):
          randInit = random.randint(0, img.shape[1] - bb_img_orig.shape[1])
        
        # crop image to adjust width to portrait width
        cropped_img = img[img.shape[0]-bb_img_orig.shape[0]:img.shape[0], randInit:randInit+bb_img_orig.shape[1], :]

        # blend images
        foreground = cv2.bitwise_and(bb_img_orig, bb_img_orig, mask = bb_inv_mask)
        background = cv2.bitwise_and(cropped_img, cropped_img, mask = bb_mask)
        out = cv2.bitwise_or(foreground, background)
        output.append(out)

        # combine with entire original video texture image
        img[img.shape[0]-bb_img_orig.shape[0]:img.shape[0], randInit:randInit+bb_img_orig.shape[1], :] = out
        img = img[:, randInit:randInit+bb_img_orig.shape[1], :]



        cv2.imwrite(output_folder + '/' + imgName  +'_'+str(counter)+".jpg", img )
        counter += 1


    # create GIF

    # writeGif("output/gif/images.gif",output,duration=0.3,dither=0)
    # name_gif = "output"
    # with imageio.get_writer('~/output.gif') as writer:
    #   for out in output:
    #     image = imageio.imread(out)
    #     writer.append_data(image)

    # add original image to video images
    # downsample whichever has greater height, and keep aspect ratio
    
    #create a video writer
    # writer = cvCreateVideoWriter(filename, -1, fps, frame_size, is_color=1)
    # #and write your frames in a loop if you want
    # cvWriteFrame(writer, frames[i])

    # images = []
    # for filename in output:
    #     images.append(imageio.imread(filename))
    # imageio.mimsave('/path/to/movie.gif', images)
    # if width 



       
def main(image_files, output_folder, resize=False):
    """Generate an HDR from the images in the source folder """

    # Print the information associated with each image -- use this
    # to verify that the correct exposure time is associated with each
    # image, or else you will get very poor results
    #print "{:^30} {:>15}".format("Filename", "Exposure Time")
    #print "\n".join(["{:>30} {:^15.4f}".format(*v)
                     #for v in zip(image_files, exposure_times)])

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]
    #print len(img_stack)
    
    if any([im is None for im in img_stack]):
        raise RuntimeError("One or more input files failed to load.")

    # Subsampling the images can reduce runtime for large files
    if resize:
        img_stack = [img[::4, ::4] for img in img_stack]

    #log_exposure_times = np.log(exposure_times)
    #hdr_image = computeHDR(img_stack, log_exposure_times)
    #cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)
    
    inputPath = SRC_FOLDER
    entirePhoto_outputPath = FACE_DETECTION_ENTIRE_PHOTO_FOLDER
    faceCrops_outputPath = SRC_FOLDER_CROP_FACES
    faceDetectorCode.faceDetection(inputPath, entirePhoto_outputPath, faceCrops_outputPath)
    #cv2.imwrite("output.jpg",img_stack[2])
    #cv2.waitKey(0)
    
    
    
    #code to read face crop and corresponding face crop mask images:
    src_contents = os.walk(SRC_FOLDER_CROP_FACES)
    dirpath, _, fnames = src_contents.next()

    image_dir = os.path.split(dirpath)[-1]
    face_crop_image_files = sorted([os.path.join(dirpath, name) for name in fnames])
    
    src_contents = os.walk(SRC_FOLDER_CROP_FACES_MASK)
    dirpath, _, fnames = src_contents.next()
    
    face_crop_mask_image_files = sorted([os.path.join(dirpath, name) for name in fnames])
    
    #print face_crop_image_files, face_crop_mask_image_files
    
    
    modifyBackground(face_crop_image_files,face_crop_mask_image_files)
    print "Done!"



if __name__ == "__main__":
    """Generate an HDR image from the images in the SRC_FOLDER directory """

    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = src_contents.next()

    image_dir = os.path.split(dirpath)[-1]
    #output_dir = os.path.join(OUT_FOLDER, image_dir)

    #try:
        #os.makedirs(output_dir)
    #except OSError as exception:
        #if exception.errno != errno.EEXIST:
            #raise

    print "Processing '" + image_dir + "' folder..."

    image_files = sorted([os.path.join(dirpath, name) for name in fnames])
    output_dir = OUT_FOLDER
    main(image_files, output_dir, resize=False)