import numpy as np
import os, fnmatch
import sys
#from SPGCam_lib import *
sys.path.append('./imread')

def list_files(dir,pattern,maxNumOfItems=0):
    listOfFiles = os.listdir(dir)

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            try:
                files = files + ',' + entry
            except:
                files = files = entry
    lines = files.split(',')

    # if user request for a max number of items
    if maxNumOfItems!=0:
        if len(lines)>maxNumOfItems:
            return lines[0:maxNumOfItems]
        else:
            return lines
    return lines

def read_raw(dir,file, width, height):
    """
    This function reads a RAW image as a Numpy array
    """
    try:
        files = list_files(dir,file)
    except:
        files=file
    image = np.zeros([width,height,len(files)],dtype=np.float32)
    for i in range(len(files)):
        im_dummy = np.fromfile(dir+files[i],dtype='>i2')
        im_dummy = im_dummy.reshape([width,height])
        image[:,:,i] = im_dummy.astype(np.float32)
    if len(files)>1:
        print('read ',len(files),'images')
    return image

def read_raw_16(dir,file, width, height, maxNumOfItems=0):
    """
    This function reads RAW images from GUIMAG as a Numpy array.
    Images size can be either 16 (no accumulated) or 32 bits (accumulated)
    """
    print('Folder found?',os.path.isdir(dir),dir)

    files = list_files(dir,file)
    files.sort()
    image = np.zeros([width,height,len(files)],dtype=np.float32)
    for i in range(len(files)):
        try: #Try 16 bits
            im_dummy = np.fromfile(dir+'/'+files[i],dtype='<i2')
            # im_dummy = np.fromfile(os.path.join(dir, files[i]),dtype='<i2')
            im_dummy = im_dummy.reshape([width,height])
            image[:,:,i] = im_dummy.astype(np.float32)
        except: #Try 32 bits
            #im_dummy=getHeader(dir+'/'+files[i])
            im_dummy=np.frombuffer(im_dummy, dtype='<i4')
            im_dummy = im_dummy.reshape([width,height])
            image[:,:,i] = im_dummy.astype(np.float32)
    if len(files)>1:
        print('read ',len(files),'images')
    return image


def imread(dir,file_in, file_out, width, height,OS='windows'):
    """
    This function executes 'imread12bpp' to convert the 12bpp output image
    of the camera to a RAW image
    """
    file_in_i=dir+file_in
    file_out_i=dir+file_out
    if OS == 'windows':
        command='imread12bpp.exe' + ' -i ' + file_in_i + ' -o ' + file_out_i \
        + ' -w ' + str(width) + ' -h ' + str(height)
    elif OS == 'OS X':
        command='imread/./imread' + ' -i ' + file_in_i + ' -o ' + file_out_i \
        + ' -w ' + str(width) + ' -h ' + str(height)
    else:
        print,'Unknown system (imread)'
    print('IMREAD',command)
    os.system(command)

def delFiles(dir,file,OS='windows'):

    if OS == 'windows':
        command='del ' + dir + '\\' + file
    elif OS == 'OS X':
        command='rm ' + dir + '/' + file
    else:
        print,'Unknown system (delFiles)'

    print('delFiles',command)
    os.system(command)
    print('delFiles:: done')
