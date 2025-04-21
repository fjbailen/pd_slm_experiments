import os
import sys
import gzip
import numpy as np
from astropy.io import fits
from scipy.io import readsav
from scipy.interpolate import interp1d
sys.path.append('../')

def create_folder(savefold):
    """
    Creates folder in a given path. If existing, displays a message.
    """
    try:
        os.mkdir(savefold)
    except:
        print(savefold+' already created')

def read_fits(Iname):
    """
    Reads FITS image and imports it as numpy array
    """
    num_im=0
    with fits.open(Iname,\
    mode='denywrite',do_not_scale_image_data=True,memmap=True) as hdul:
        hdul.info()
        data = hdul[0].data #Stokes order: I,V,Q,U
    return data

def read_sav(Iname):
    """
    Reads SAV file and imports it as numpy array
    """
    data=readsav(Iname)
    return data

def read_gzfits(Iname):
    """
    Reads FITS image compressed as .gz and imports it as numpy array
    """
    num_im=0
    with gzip.open(Iname,'rb') as im:
        with fits.open(im,\
        mode='denywrite',do_not_scale_image_data=True,memmap=True) as hdul:
            hdul.info()
            print(hdul[0].header)
            data = hdul[0].data #Stokes order: I,V,Q,U
    return data

def save_fits(A,fname):
    hdu=fits.PrimaryHDU(A)
    hdul=fits.HDUList([hdu])
    try:
        hdul.writeto(fname)
    except OSError:
        print(fname+' already exists. Overwriting...')
        os.remove(fname)
        hdul.writeto(fname)

def save_fits2(A,b,fname):
    #A: multidimensional array with the images
    primary_hdu=fits.PrimaryHDU(A)


    #b: one dimensional array with defocuses
    second_hdu=fits.ImageHDU(b)

    #Save fits
    hdul=fits.HDUList([primary_hdu, second_hdu])
    try:
        hdul.writeto(fname)
    except OSError:
        print(fname+' already exists. Overwriting...')
        os.remove(fname)
        hdul.writeto(fname)

def read_file(fname):
    """
    Read txt file and export the data as a numpy matrix
    """
    return np.genfromtxt(fname,unpack=False,dtype=None,encoding='utf-8')

def from64to129(A):
    """
    Converts 65x65 matrix into 129x129 matrix
    """
    A=np.concatenate((np.flip(A,axis=1),A),axis=1)
    A=np.concatenate((np.flip(A,axis=0),A),axis=0)
    A=np.delete(A,64,axis=0)
    return np.delete(A,64,axis=1)

def from1Dto2D(atilde):
    """
    Interpolates data to make the PSF radially symmetric from 1D to 2D
    """
    def centeredDistanceMatrix(n):
    # make sure n is odd
        n=int(2*n)
        x,y = np.meshgrid(range(n),range(n))
        return np.sqrt((x-(n/2)+1)**2+(y-(n/2)+1)**2)

    def arbitraryfunction(d,y,n):
        x = np.arange(n)
        f = interp1d(x,y,bounds_error=False,fill_value=0)
        return f(d.flat).reshape(d.shape)

    n = atilde.shape[0]
    d = centeredDistanceMatrix(n)
    f = arbitraryfunction(d,atilde,n)
    return f[:-1,:-1]

def vect_app2(profile1D):
    l = 2*len(profile1D)
    critDim=int((l**2 /2.)**(1/2.))
    a = np.arange(critDim)**2
    r2D = np.sqrt(a[:,None] + a)
    out = np.take(profile1D,(l+r2D).astype(int))
    return out

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gaussian_noise(SNR,image):
    row,col = image.shape
    mu  = np.mean(image)
    sigma= mu*SNR
    gauss = np.random.normal(0,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy
