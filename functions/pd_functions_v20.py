"""
Same as pd_functions_v16, except for the fact that now admits
a phase diversity different from a defocus. In particular,
the parameter a_d can be now an array. If so, it must contain
the Zernike coefficients of the phase diversity in order.

Changes with respect to pd_functions_v16
-'pupil' and 'wavefront' now differentiate between a_d being a scalar or an array
-Normalization is corrrect in read_image, so the focused and defocused
images are normalized independently, each one by their own mean.
-low_f is increased up to 0.2 and a second low filtering is applied
to the noise filter in order to elliminate isolated peaks

Changes with respect to pd_functions_v17
-prepare_PD now uses the Sicairos method (dftreg) to
align the FFTs with subpixel accuracy before feeding
the PD algorithm (modification made on 2023/3/10)
-we compute the optimized merit function when calling to "object_estimate"
(modification made on 2023/3/16)
-noise_filter is calculated in the 4 corners of the FFT image
-gamma is calculated with non-apodized images, like in v1.6
-A correction in filter_sch is applied. The denominator was calculated
using gamma1(=1) instead of the calculated gamma.
-Tip/tilt correction is applied ONLY to defocused images in case Jmin<4
-low_f can be introduced as a parameter in 'object_estimate' and in 'loop_opt'
"""
#from skimage.morphology import square, erosion
from matplotlib import pyplot as plt
from PIL import Image as im
from photutils.aperture import CircularAperture
from scipy.ndimage import median_filter
from scipy import ndimage, misc
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.optimize import minimize, minimize_scalar
from astropy.convolution import convolve
import numpy as np
import zernike as zk
import math_func2 as mf
import plots_func2 as pf
import sys
import os
import time
import scipy
import shift_func as sf
from astropy.io import fits
try:
    import pyfftw
    flag_pyfftw=1
except:
    flag_pyfftw=0


#PHI parameters
wvl=633e-9 #Wavelength [m]
gamma1=1 #Gamma defined by Lofdahl and Scharmer 1994
gamma2=0 #Gamma factor defined by Zhang 2017 to avoid divergence of "Q"
N=924 #824
fnum=119 # f-number
Delta_x=12e-6 #Size of the pixel [m]
nuc=1/(wvl*fnum) #Critical frequency (m^(-1))
inc_nu=1/(N*Delta_x)
R=(1/2)*nuc/inc_nu #Pupil radius [pixel]
nuc=2*R #critical frequency (pixels)
nuc-=1#1 or 2 #To avoid boundary zeros when calculating the OTF (Julian usa 1)



def sampling(N=N):
    """
    This function creates a grid of points with NxN dimensions for calling the
    Zernike polinomials.
    Output:
        X,Y: X and Y meshgrid of the detector
    """

    if N%2 != 0: #If N is odd (impar)
        print('Error in function "sampling": Number of pixels must be even!')
        x=np.linspace(-round(N/2),round(N/2),N)
        #sys.exit()
    else: #If N is even (par)
        x=np.linspace(-N/2,N/2,N)
        #x=np.linspace(-N/2,N/2,N+1)
        #x=x[:-1]

    y=np.copy(x)
    X,Y=np.meshgrid(x,y)

    return X,Y

def sampling2(N=N,R=R):
    """
    It returns RHO and THETA with the sampling of the pupil carried out
    in fatima.py. RHO is already normalized to the unity.
    """
    r=1
    x=np.linspace(-r,r,2*int(R))
    y=np.copy(x)
    [X,Y]=np.meshgrid(x,y)
    RHO = np.sqrt(X**2+Y**2)
    THETA = np.arctan2(Y, X)
    RHO0=np.zeros((N,N))
    THETA0=np.copy(RHO0)
    RHO0[N//2-int(R):N//2+int(R),N//2-int(R):N//2+int(R)]=RHO
    THETA0[N//2-int(R):N//2+int(R),N//2-int(R):N//2+int(R)]=THETA
    return RHO0,THETA0

def cir_aperture(R=R,N=N,ct=None):

    #Offset to be subtracted to center the circle
    if ct==None:
        if N%2 != 0: #If N is odd (impar)
            #print('Number of pixels should be an even integer!\n')
            ct = 1/2
        else: #If N is even (par)
            ct = -1/2#1/2

    N = np.int(N)
    R = np.int(R)
    A = CircularAperture((N/2-ct,N/2-ct),r=R) #Circular mask (1s in and 0s out)
    A = A.to_mask(method='exact') #Mask with exact value in edge pixels
    #A = CircularAperture((N/2-ct,N/2-ct),r=R-1)
    #A = A.to_mask(method='center')

    A = A.to_image(shape=(N,N)) #Conversion from mask to image
    return A



def pmask(nuc=nuc,N=N):
    """
    Mask equal to 1 where cir_aperture is 0
    """
    pmask=np.where(cir_aperture(R=nuc,N=N,ct=0)==0,1,0)
    #pmask=1-cir_aperture(R=nuc,N=N)
    #pmask=np.where(pmask<0,0,pmask)
    return pmask

def aperture(N,R,cobs=0,spider=0):
    """
    This function calculates a simple aperture function that is 1 within
    a circle of radius R, takes and intermediate value between 0
    and 1 in the edge and 0 otherwise. The values in the edges are calculated
    according to the percentage of area corresponding to the intersection of the
    physical aperture and the edge pixels.
    http://photutils.readthedocs.io/en/stable/aperture.html
    Input:
        N: 1D size of the detector
        R: radius (in pixel units) of the pupil in the detector
        cobs: central obscuration (as percentage of radius)
        spider: width of spider arms (in pixels)
    Output:
        A: 2D array with 0s and 1s
    """
    A=cir_aperture(R=R,N=N)

    #If central obscuration:
    if (cobs != 0):
        if N%2 != 0:
            #print('Number of pixels should be an even integer!\n')
            ct = 0.
        else:
            ct = 1/2
        B=CircularAperture((N/2-ct,N/2-ct),r = R*cobs/100.)
        B=B.to_mask(method='exact') #Mask with exact value in edge pixels
        B=B.to_image(shape=(N,N)) #Conversion from mask to image
        A=A-B
        A = np.where(A<=0,0,A)

    #If spider:
    if (spider != 0):
        C=array((N*10,N*10))
        M = N*10 + 10
        S = R*spider/10.
        C[int(M/2-S/2):int(M/2+S/2),:] = 1.0
        C[:,int(M/2-S/2):int(M/2+S/2)] = 1.0
        nC = imresize(C, (N, N))#, interp="bilinear")
        nC = nC / np.max(nC)
        nC=np.where(nC<=0,0,nC)
        A = A - nC
        A = np.where(A<=0,0,A)
    return A

def aperture_easy(X,Y):
    """
    This function calculates a simple aperture function that is 1 within
    a circle of radius R and 0 otherwise.
    Input:
        X,Y: meshgrid with the coordinates of the detector ('sampling.py')
        R: radius (in pixel units) of the pupil in the detector
    Output:
        A: 2D array with 0s and 1s
    """
    A=X**2+Y**2<=R**2
    return A

def pupil(a,a_d,RHO,THETA,A):
    """
    This function calculates the generalized pupil function of a circular
    aperture for an incident wavefront with an aberration given by the
    Zernike coefficients following Noll's order.
    The aperture function is given by A ('aperture.py').
    Input:
        a: 1D array with Zernike coefficients (ordered following Noll's order
           The first element,a[0], is the Piston, a[1] and a[2] are Tilts
           in the X and Y direction ...)
        a_d: Defocus Zernike coefficient for defocused image (float or int)
            or array with all the aberrations introduced by the defocused image
        RHO,THETA: meshgrid with polar coordinates (rho normalised to 1)
        A: aperture 2D array ('aperture')
    Output:
        p: 2D complex array describing the pupil function
    """
    try:
        Jmax=len(a) #Max. number of Zernikes to be employed
    except TypeError:
        print('Error in "pupil" function: "a" must be a Numpy array or a list')
        sys.exit()

    #Phase induced during PD

    if isinstance(a_d, (float,int)): #If float, then induce a defocus a_d
        phi=a_d*zk.zernike(0,2,RHO,THETA)#diversity phase
    else:
        #If array, then get zernike coefficients from a_d
        Jmax_div=len(a_d)
        jj=0
        phi=0*RHO #To have an array filled with zeroes
        while jj<Jmax_div:
            jj+=1
            phi+=a_d[jj-1]*zk.zernikej_Noll(jj,RHO,THETA)

    #Wavefront due to a coefficients
    j=0
    while j<Jmax:
        j+=1
        phi+=a[j-1]*zk.zernikej_Noll(j,RHO,THETA)
    phi=A*phi
    p=A*np.exp(1j*phi)
    return p

def wavefront(a,a_d,RHO,THETA,A,R=R,N=N):
    """
    This function returns the wavefront map given a set of Zernike
    coefficients. It works similarly than pupil.py
    """
    try:
        Jmax=len(a) #Max. number of Zernikes to be employed
    except TypeError:
        print('Error in "pupil" function: "a" must be a Numpy array or a list')
        sys.exit()
    #Phase diversity
    if isinstance(a_d, (float,int)): #If float, then induce a defocus a_d
        phi=a_d*zk.zernike(0,2,RHO,THETA)#diversity phase
    else: #If array, then get zernike coefficients from a_d
        Jmax_div=len(a_d)
        jj=0
        phi=0*RHO #To have an array filled with zeroes
        while jj<Jmax_div:
            jj+=1
            phi+=a_d[jj-1]*zk.zernikej_Noll(jj,RHO,THETA)
    j=0
    while j<Jmax:
        j+=1
        phi+=a[j-1]*zk.zernikej_Noll(j,RHO,THETA)
    phi=A*phi #Calcular phi en todo X,Y para luego hacer esto
                              #puede no ser lo más óptimo
    #Subframing to cover only the pupil
    xc=int(N/2)
    x0=int(xc-R+1)
    xf=int(xc+R+1)

    phi=phi[x0:xf,x0:xf]

    return phi

def PSF(a,a_d,RHO,THETA,ap):
    """
    This function calculates the PSF of a circular aperture for an incident
    wavefront with a wavefront aberration given by the Zernike coefficients.
    The aperture function is defined in 'aperture.py'
    Input:
        a,a_d,RHO,THETA,A: defined in 'pupil' function
    Output:
        psf: 2D real array representing the PSF
    """
    #Generalized complex pupil function
    p=pupil(a,a_d,RHO,THETA,ap)
    #PSF
    psf=np.abs(mf.fourier2(p))**2 #Non normalized PSF (normalization made in OTF)
    return psf

def OTF(a,a_d,RHO,THETA,ap,norm=None,nuc=nuc,N=N,K=2):
    """
    This function calculates the OTFs of a circular aperture for  incident
    wavefronts with aberrations given by a set of Zernike coefficients.
    The OTF is calculated as the autocorrelation of the pupil.
    Input:
        a,a_d,RHO,THETA,A: defined in 'pupil.py'
            a_d can be an array with shape K, where K is the number
            of images with different defocuses. In such a case, a_d
            contains the defocus coefficient of each image and the program
            returns an multidimensional array of OTFs
        norm:{None,True}, optional. 'True' for normalization purpose. Default
         is None
    Output:
        otf: 2D complex array representing the OTF if len(a_d)!=K or
            a 3D complex array whose 3rd dimenstion indicates the  OTF of
            of each defocused image.
        norm: normalization factor of the OTF

    """
    #If a_d is a number or a vector containing the aberrations of the PD plate

    if isinstance(a_d, (float,int)) or len(a_d)>K:
        #Pupil
        p=pupil(a,a_d,RHO,THETA,ap)
        #OTF
        norma_otf,otf=mf.corr(p,p,norma=True)
        if norm==True:
            norma=norma_otf
            #norma=np.max(np.abs(otf)[:])
            otf=otf/norma #Normalization of the OTF
        else:
            norma=1
    #If a_d is an array containing K diversities
    elif len(a_d)==K:
        otf=np.zeros((RHO.shape[0],RHO.shape[0],K),dtype='complex128')
        norma=np.zeros(K)

        for i in range(K):
            #Pupil.
            if i==0: #For the focused image, do not consider tip/tilt
                afocus=np.copy(a)
                afocus[:3]=0
                p=pupil(afocus,a_d[i],RHO,THETA,ap)
            else: #For other images, consider all Zernike coefficients
                p=pupil(a,a_d[i],RHO,THETA,ap)

            #OTF
            norma_otf,otf[:,:,i]=mf.corr(p,p,norma=True)
            if norm==True:
                norma[i]=norma_otf
                #norma=np.max(np.abs(otf)[:])
                otf[:,:,i]=otf[:,:,i]/norma[i] #Normalization of the OTF
            else:
                norma[i]=1
    return otf,norma


def OTF_circular():
    """
    This function returns the analytical expression of the OTF of a circular
    aperture with no aberrations.
    Output:
        otf: 2D complex array representing the OTF of a circular aperture
    """
    nu=np.linspace(-N/2,N/2,N)
    nu_norm=nu/nuc
    mask=np.abs(nu_norm**2)<=1
    otfc_out=np.zeros(N)
    otfc_in=(2/np.pi)*(np.arccos(np.absolute(nu_norm*mask))-\
    np.absolute(nu_norm*mask)*np.sqrt(1-(nu_norm*mask)**2))
    otfc=np.where(mask,otfc_in,otfc_out)
    return otfc

def convPSF(I,a,a_d,RHO,THETA,ap,norm=None,nuc=nuc,N=N):
    """
    This function calculates the convolution of an image with the PSF
    of IMaX+. 'I' must be a Numpy array of type 'float64'
    Input:
        I: (real) 2D numpy array (image)
        a,a_d,RHO,THETA,A: defined in 'pupil.py'
    Output:
        d: real 2D numpy array representing the convolution of I and the system
    """
    if I.dtype != 'float64':
        print(I.dtype,'Error: I must be a Numpy array of type float64')
        sys.exit()
    #Fourier transform of the image
    O=fft2(I)
    O=ifftshift(O)

    #Aberrated image
    otf,norma=OTF(a,a_d,RHO,THETA,ap,norm=norm,nuc=nuc,N=N)
    D=O*otf

    d=ifftshift(D)
    d=ifft2(d).real
    return d

def Zkfactor(k,Ok,Hk,Q,gamma=gamma1):
    """
    Factor Zk defined in Paxman 1992 (eq. C6) needed to calculate
    the derivative of the merit function (eqs. C1-C5)
    Input:
        k:index 'k' of Zk (k<= #of different defocuses)
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Zk: 2D array with the Zk factor
    """
    Qinver=Qinv(Hk,gamma=gamma)
    sum1=Qinver**2*np.sum(Ok*np.conj(Hk),axis=2)*np.conj(Ok[:,:,k])
    sum2=np.abs(np.sum(Ok*np.conj(Hk),axis=2))**2*gamma[k]*np.conj(Hk[:,:,k])
    Zk=Q**4*(sum1-sum2)
    return Zk


def Qfactor(Hk,gamma=gamma1,nuc=nuc,N=N):
    """
    Q factor defined in Lofdahl and Scharmer 1994 for construction of the merit
    function
    Input:
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Q: 2D complex numpy array representing the Q factor
    """
    #np.seterr(divide='ignore')
    Q=1/np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2))
    Q=np.nan_to_num(Q, nan=0, posinf=0, neginf=0)
    Q=Q*cir_aperture(R=nuc,N=N,ct=0)
    return Q


def Qinv(Hk,gamma=gamma1,nuc=nuc,N=N):
    """
    Inverse of Q. Defined for test purposes
    Input:
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Qinv:
    """
    Qinv=np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2))
    Qinv=Qinv*cir_aperture(R=nuc,N=N,ct=0)
    return Qinv

def Ffactor(Q,Ok,Hk,gamma=gamma1,nuc=nuc,N=N):
    """
    F factor defined by Lofdahl and Scharmer 1994 (Eq. 5) in the general
    form of Paxman 1992 (Eq. 19). Gamma is added, too.
    Input:
        Q: Q returned in Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Zk: 2D array with the Zk factor
    """

    return Q**2*np.sum(gamma*Ok*np.conj(Hk),axis=2)

def smooth_filt(array,size=3):
    """
    Function that smoothes a 2D array using ndimage.uniform_filter
    """
    #kernel = np.ones((size,size))
    #return convolve(array, kernel)
    return ndimage.uniform_filter(array, size=size)


def noise_power(Of,filterfactor=1.5,nuc=nuc,N=N):
    """
    Average level of noise power of the image. Based on Bonet's program
    'noise_level.pro'. Noise is calculated on 4 quadrants of the
    FFT of the image, beyond the critical frequency, to elliminate
    the dreadful cross (horizontal and vertical spurious signal in the FFT)
    that appears because of the finite Nyquist frecuency of the detector.
    """
    #Circular obscuration mask to calculate the noise beyond the critical freq.
    cir_obs=pmask(nuc=nuc,N=N)


    #Calculation of noise
    #power=np.sum(np.abs(Of)**2*cir_obs)/np.sum(cir_obs)

    #1st quadrant
    x2=int(np.floor(N/2-nuc*np.sqrt(2)/2))
    power=np.sum((np.abs(Of)**2*cir_obs)[0:x2,0:x2])/np.sum(cir_obs[0:x2,0:x2])
    #2nd quadrant
    x3=N-x2
    power+=np.sum((np.abs(Of)**2*cir_obs)[x3:N,0:x2])/np.sum(cir_obs[x3:N,0:x2])
    #3rd quadrant
    power+=np.sum((np.abs(Of)**2*cir_obs)[0:x2,x3:N])/np.sum(cir_obs[0:x2,x3:N])
     #4th quadrant
    power+=np.sum((np.abs(Of)**2*cir_obs)[x3:N,x3:N])/np.sum(cir_obs[x3:N,x3:N])

    #To obtain a more conservative filter in filter_sch
    power=filterfactor*power
    return power

def filter_sch(Q,Ok,Hk,low_f=0.2,gamma=gamma1,nuc=nuc,N=N):
    """
    Filter of the Fourier transforms of the focused and defocused
    object (Eqs. 18-19 of Lofdahl & Scharmer 1994). Based on
    filter_sch.pro (Bonet's IDL programs)
    Input:
        Q: Q returned in Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        filter:2D array (float64) with filter
    """

    denom=np.abs(Ffactor(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N))**2\
    *Qinv(Hk,gamma=gamma,nuc=nuc,N=N)**2


    filter=noise_power(Ok[:,:,0],nuc=nuc,N=N)/smooth_filt(denom)
    filter=(filter+np.flip(filter))/2
    filter=1-filter
    filter=np.where(filter<low_f,0,filter)
    filter=np.where(filter>1,1,filter)
    #filter=erosion(filter, square(3)) #To remove isolated peaks (also decreases
                                       #the area of the central region)

    filter=median_filter(filter,size=9)
    filter=smooth_filt(filter)
    filter=filter*cir_aperture(R=nuc,N=N,ct=0)
    filter=np.nan_to_num(filter, nan=0, posinf=0, neginf=0)

    #Apply low_filter again to elliminate isolated peaks
    filter=np.where(filter<low_f,0,filter)
    return filter


def meritE(Ok,Hk,Q):
    """
    Merit function 'E' for phase diversity optimization
    Ref: Lofdahl and Scharmer 1994, Eq.9
    Input:
        Q: 2D array returned by Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Ouput:
        E: 2D complex array with E factor for construction of the merit function
            or 3D complex array with the error metrics of each possible
            pair of the K images
        function
    """
    #Lofdahl error metric
    #E=Q*(Ok[:,:,1]*Hk[:,:,0]-Ok[:,:,0]*Hk[:,:,1])

    #General error metrix from Paxman merit function
    #E=Q*np.sum(Ok*np.conj(Hk),axis=2)

    #Alternative general error metric
    #E=Q*sumamerit(Ok,Hk)

    #Array with invidual error metrics
    E=arraymerit(Ok,Hk)
    if E.ndim==3:
        for i in range(E.shape[2]):
            E[:,:,i]=Q*E[:,:,i]
    elif E.ndim==2:
        E=Q*E
    return E

def derivmerit(Ok,Hk,derHk,Q,gamma=gamma1,cut=None):
    """
    Derivative of the merit function of Paxman 1992 (eq. C5)
    Input:
        Q: 2D array returned by Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
        derHk: 3D array with the derivatives of the OTF for each defocus
    """


    sumZkSk=np.zeros((Ok.shape[0],Ok.shape[1]),dtype='complex128') #Initialization of the sum
    for k in range(Ok.shape[2]):
        sumZkSk+=Zkfactor(k,Ok,Hk,Q,gamma=gamma)*derHk[:,:,k]#1st term of eq. C5


    #Image domain
    sumZkSk=ifftshift(sumZkSk)
    N=sumZkSk.shape[0]
    sumzksk=N*N*ifft2(sumZkSk)
    sumzksk-=np.mean(sumzksk)

    #Parseval theorem
    n=sumzksk[cut:-cut,cut:-cut].shape[0]
    deriv=sumzksk[cut:-cut,cut:-cut]
    #deriv+=np.conj(deriv) #Second term of eq. C5

    deriv=2*np.real(deriv)#1st + 2nd term = 2*real(1st term)
    deriv=np.sum(deriv)/n**2

    return deriv

def sumamerit(A,B):
    """
    This function returns the double summatory from j=1 to K-1
    and from k=j+1 to K defined by Paxman (Eq. B10)to calculate the merit
    function.
    Input:
        A,B: 3D arrays of shape (N,N,K) containing Ok and Hk/derHk
    """
    K=A.shape[2] #Number of diversities (including focused image)
    suma=0 #Initialization of E
    for j in range(K-1):
        for k in range(j+1,K):
            suma+=A[:,:,j]*B[:,:,k]-A[:,:,k]*B[:,:,j]
    return suma

def arraymerit(A,B):
    """
    This function returns calculates the individual error
    functions of Eq. 20 in Paxman and it stores them in
    an array of shape (N,N,K).
    Input:
        A,B: 3D arrays of shape (N,N,K) containing Ok and Hk or derHk
    """
    K=A.shape[2] #Number of diversities (including focused image)
    nk=int((K-1)*K/2)
    arraymerit=np.zeros((A.shape[0],A.shape[1],nk),dtype='complex128') #Initialization of E
    i=-1
    for j in range(K-1):
        for k in range(j+1,K):
            i+=1
            arraymerit[:,:,i]=A[:,:,j]*B[:,:,k]-A[:,:,k]*B[:,:,j]
    return arraymerit

def meritL(E):
    """
    Merit function "L" defined by Löfdahl & Schamer 1994 (Eq. 8).
    """
    L=np.sum(np.abs(E)**2)#-np.sum(np.sum(np.abs(Ok)**2,axis=2))
    return L

def merite(E):
    """
    Inverse transform of 'E' ('meritE' function).
    If E is a 3D array, it returns also a 3D array with the IFFT of each metric.
    Input:
        E: 2D complex array with E factor for construction of the
            classical (Löfdahl & Scharmer) merit function
            or 3D complex array with the error metrics for each possible
            combination of pairs of the K images
    Output:
        e: IFFT of E (2D array or 3D depending of the shape of E)
    """
    if E.ndim==3: #Array with individual merit functions of K diverities
        e=np.zeros(E.shape)
        for i in range(E.shape[2]):
            e0=ifftshift(E[:,:,i])
            e0=ifft2(e0)
            n=E.shape[0]
            e0=n*n*e0.real
            e[:,:,i]=e0-np.mean(e0)#We substract the mean value (Bonet e_func.pro (CORE2007))
    elif E.ndim==2:
        e=ifftshift(E)
        e=ifft2(e)
        n=E.shape[0]
        e=n*n*e.real
        e-=np.mean(e)#We substract the mean value (Bonet e_func.pro (CORE2007))
    return e

def meritl(e,cut=None):
    """
    Merit function "L" defined by Lofdahl and Scharmer 1994.
    If 'e' is a 3D array, the merit function is generalized to work for
    K images with different defocuses
    """
    cutx=cut
    cuty=cut
    L=0
    if e.ndim==3: #If dim=3: sum individual merit functions
        for i in range(e.shape[2]):
            L+=np.sum(np.abs(e[cutx:-cuty,cutx:-cuty,i])**2)/e.shape[0]**2
    elif e.ndim==2:
        if cut!=None:
            L=np.sum(np.abs(e[cutx:-cuty,cutx:-cuty])**2)/e.shape[0]**2
        else:
            L=np.sum(np.abs(e[:,:])**2)
    return L


def derHj2(zj,p,norma):
    """
    Derivative of the OFT with respect to Zernike coefficient 'a_j' using
    the correlation definition of math_func2
    Ref: Paxman et al 1992, Eq. C14
    Inputs:
        zj: Zernike 'j' in Noll's basis
        p: generalized pupil function
    """
    K=p.shape[2]
    derivHj=np.zeros(p.shape,dtype='complex128')
    for i in range(K):
        C1=mf.corr(p[:,:,i],zj*p[:,:,i])
        C2=mf.corr(zj*p[:,:,i],p[:,:,i]) #C2=np.conj(np.flip(C1))
        corr1_corr2=C1-C2 #Substraction
        derivHj[:,:,i]=1j*(corr1_corr2)/norma[i]
    return derivHj

def derEj(derHk,Ok,Hk,Q,norma,gamma=gamma1):
    """
    Derivative with respect to a_j of the Merit function 'E'
    for phase diversity optimization
    Refs: 1) Lofdahl and Scharmer 1994, Eq.12
          2) Paxman et al 1992, Eq. C14
    Input:
        Q: 2D array returned by Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
        derHk: 3D array with the derivatives of the OTF for each defocus

    Output:
        2D complex array with the derivative of E for construction of the
        classical (Löfdahl & Scharmer) merit function
        or 3D complex array with the derivatives of each error function of the
        combination of pairs of the K images
    """
    #Derivate of Q
    derQ=-Q**3*np.sum(gamma*np.real(np.conj(Hk)*derHk),axis=2)

    #Derivative of Lofdahl error function
    #derivE=Ok[:,:,1]*(Q*derHk[:,:,0]+derQ*Hk[:,:,0])\
    #-Ok[:,:,0]*(Q*derHk[:,:,1]+derQ*Hk[:,:,1])

    #Derivative of general error metric from Paxman merit function
    #derivE=derQ*np.sum(Ok*np.conj(Hk),axis=2)+Q*np.sum(Ok*np.conj(derHk),axis=2)

    #Derivative of the alternative error metric
    #derivE=derQ*sumamerit(Ok,Hk)+Q*sumamerit(Ok,derHk)

    #Array containing the derivative of each error function
    if Ok.ndim==3:
        K=Ok.shape[2]
        nk=int((K-1)*K/2)
        derivE=np.zeros((Ok.shape[0],Ok.shape[1],nk),dtype='complex128')
        for i in range(derivE.shape[2]):
            derivE[:,:,i]=derQ*arraymerit(Ok,Hk)[:,:,i]+Q*arraymerit(Ok,derHk)[:,:,i]
    elif Ok.ndim==2:
        derivE=derQ*arraymerit(Ok,Hk)+Q*arraymerit(Ok,derHk)
    return derivE


def derej(derE):
    """
    Inverse transform of the derivative of 'E' with respect to a_j.
    Input:
        derE: 2D complex array with the derivative of E for construction of the
            classical (Löfdahl & Scharmer) merit function
            or 3D complex array with the derivative of E for each possible
            combination of pairs of the K images
    Output:
        derej: IFFT of derE (2D array or 3D depending of the shape of E)
    """
    n=derE.shape[0]
    if derE.ndim==3: #Array with individual merit functions of K diverities
        dere=np.zeros(derE.shape)
        for i in range(derE.shape[2]):
            dere0=ifftshift(derE[:,:,i])
            dere0=n*n*ifft2(dere0).real
            dere[:,:,i]=dere0-np.mean(dere0)
    elif derE.ndim==2: #If dim=2, no multidimensional array, just a n x n image
        dere=ifftshift(derE)
        dere=n*n*ifft2(dere).real
        dere-=np.mean(dere)
    return dere

def Aij(derivEi,derivEj,cut=None):
    """
    Elements of the A matrix (Lofdahl and Scharmer 1994, Eq 15)
    """
    Aij=derivEi*np.conj(derivEj)
    cutx=cut
    cuty=cut

    if cut!=None:
        Aij=np.sum(Aij[cutx:-cuty,cutx:-cuty])/Aij[cutx:-cuty,cutx:-cuty].shape[0]**2
    else:
        Aij=np.sum(Aij[:])/Aij.shape[0]**2
    return Aij.real

def b_i(derivEi,E,cut=None):
    """
    Elements of the b vector (Lofdahl and Scharmer 1994, Eq 16)
    """
    bi=derivEi*np.conj(E)
    cutx=cut
    cuty=cut
    if cut!=None:
        bi=np.sum(bi[cutx:-cuty,cutx:-cuty])/bi[cutx:-cuty,cutx:-cuty].shape[0]**2
    else:
        bi=np.sum(bi[:])/bi.shape[0]**2
    return bi.real

def loop_opt(tol,Jmin,Jmax,w_cut,maxnorm,maxit,a0,a_d,RHO,THETA,ap,Ok,\
disp=True,cut=None,method='svd',gamma=gamma1,nuc=nuc,N=N,ffolder='',K=2,low_f=0.2):
    """
    Function that carries out the iterative loop for optimizing the aberration
    coefficients in the Phase Diversity process. It employs the focused and
    defocused images of the object and returns an array with the optimum
    aberration coefficients.
    Inputs:
        tol: tolerance criterion to stop iterations (change in the norm of a)
        Jmin: lowest Noll's Zernike index to be fitted
        Jmax: upper limit of Noll's Zernike. The highest Zernike index fitted
            is Jmax-1.
        w_cut: SVD cutoff as a fraction of the maximum singular value
        maxnorm: max norm of a allowed in the iterative process
        maxit: maximum number of iterations allowed for the iterative process
        a0: initial guess for the Zernike indices (usually, array with zeroes)
        a_d: phase difference between images. It can be a scalar (classic
            two-images PD, or an array of length K if we use K images)
        RHO, THETA: output of 'sampling2' function
        ap: output of 'aperture' function
        Ok: output of 'prepare_PD'. Consists of an array with
            the FFT of the input PD images. The first index
            must correspond to the 'focused' image. The images should be
            arranged consistently with the phase differences of 'a_d'
        disp (True or False): to print some information on the iterative process
        cut: number of edge pixels on the images that are cropped
        method: optimization method ('svd' is highly recommended)
        gamma: output of "prepare_PD". Level of noise among images.
        nuc: critical frequency of the telescope (computed at the beginning
            of this module)
        N: size of each PD image (defined at the beginning of this module)
        ffolder: subfolder where the results will be saved (parent folder is 'txt')
        K: number of images employed (2 for focused-defocused classic PD)
        low_f: cut-off for Scharmer's optimum filter
    Output:
        a_iteration_Jmax_...:txt file with information on the merit function,
            norm of the change of Zernike coefficients and the Zernike coefficients
            at each iterative step. When subfielding is applied, only
            the information corresponding to the subfield that takes
            longer to be optimized is saved.
        a_optimized_Jmax_...: file with information of the optimized
            Zernike coefficients. If subfielding is applied, then
            one file per subfield is saved. Subfields are enumerated
            with one label (k)
    """
    print('gamma:',gamma)
    a=np.copy(a0)



    #Reference defocus for comparing purposes
    d0=8*wvl*(fnum)**2 #Defocus shift to achieve 1 lambda peak-to-valley WF error
    a_d0=np.pi*d0/(8*np.sqrt(3)*wvl*fnum**2) #Defocus coefficient [rads]
    tic=time.time()
    it=0
    norm_delta=maxnorm

    #Initialization of A and b
    A=np.zeros((Jmax-Jmin,Jmax-Jmin),dtype='float64')
    b=np.zeros((Jmax-Jmin,1),dtype='float64')

    #Initialization of the array to be saved in file
    L=0
    param=np.array([it,L,norm_delta])
    vfile=np.concatenate((param.reshape(len(param),1),a))
    file=np.copy(vfile)
    a_last=a[(Jmin-1):]+1
    a2=np.zeros((Jmax-1,1))

    #Planning for pyfftw
    if flag_pyfftw==1:
        pf=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')
        pd=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')
        zi_pf=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')
        zi_pd=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')
        zj_pf=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')
        zj_pd=pyfftw.empty_aligned((RHO.shape[1],RHO.shape[1]),dtype='complex128')

    while True:
        it+=1
        if disp is True:
            print('\n')
            print('Iteration',it)
        #Focused and defocused OTFs
        Hk,normhk=OTF(a,a_d,RHO,THETA,ap,norm=True,nuc=nuc,N=N,K=K) #Focused OTFs
        norma=normhk[0] #Norm for correlations

        #Q factor, merit function and focused and defocused pupils
        Q=Qfactor(Hk,gamma=gamma,nuc=nuc,N=N)
        noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N,low_f=low_f)


        Ok_filt=np.zeros((N,N,K),dtype='complex128')
        pk=np.zeros((N,N,K),dtype='complex128')
        for i in range(K):
            Ok_filt[:,:,i]=noise_filt*Ok[:,:,i] #Filtered FFT
            pk[:,:,i]=pupil(a,a_d[i],RHO,THETA,ap) #Pupil


        E=meritE(Ok_filt,Hk,Q)
        e=merite(E)
        L_last=L
        L=meritl(e,cut=cut)

        derivL=np.abs(L-L_last)/L
        L2=meritL(E)

        #A matrix and b vector
        A_last=np.copy(A)
        b_last=np.copy(b)

        #fig,axs=plt.subplots(1,5)
        for i in range(Jmin,Jmax):
            zi=zk.zernikej_Noll(i,RHO,THETA)
            derHk_i=derHj2(zi,pk,normhk)
            derivEi=derEj(derHk_i,Ok_filt,Hk,Q,norma,gamma=gamma)
            derivei=derej(derivEi)
            b[i-Jmin]=b_i(derivei,e,cut=cut)

            """
            if (i-Jmin)<5:
                axs[i-Jmin].imshow(derivei)
                axs[i-Jmin].set_title(i)
            if (i-Jmin)==5:
                plt.show()
                quit()
            """

            for j in range(i,Jmax):
                if i==j:
                    derivEj=derivEi
                    derivej=derivei
                else:
                    zj=zk.zernikej_Noll(j,RHO,THETA)
                    derHk_j=derHj2(zj,pk,normhk)
                    derivEj=derEj(derHk_j,Ok_filt,Hk,Q,norma,gamma=gamma)
                    derivej=derej(derivEj)

                A[i-Jmin,j-Jmin]=Aij(derivei,derivej,cut=cut)
                A[j-Jmin,i-Jmin]=A[i-Jmin,j-Jmin] #A is symmetric

        #plt.imshow(A,origin='lower',vmin=-1e-4,vmax=1e-4,cmap='seismic')
        #plt.show()
        #plt.close()
        #quit()

        #SVD decomposition
        delta_a=mf.svd_solve(A,-b,w_cut,method=method,rms_limit=maxnorm)

        #Next iteration or break the loop
        norm_last=norm_delta
        norm_delta=np.linalg.norm(delta_a)
        if disp is True:
            print('Norm delta_a:',round(norm_delta,3))

        if norm_delta>maxnorm:
            print('\nOptimization stopped: Norm of delta_a >',maxnorm)
            print('Defocus:',round(a_d/a_d0,2),' lambda')
            print('Number of iterations:',it-1)
            break
        elif it>maxit:
            if disp is True:
                print('\nMaximun number of iterations reached:',maxit)
            break

        else:
            flag=0
            a_last=np.copy(a[(Jmin-1):])
            a[(Jmin-1):]+=delta_a
            param=np.array([it,L,norm_delta])
            vfile=np.concatenate((param.reshape(len(param),1),a))
            file=np.column_stack((file,vfile))
        if tol>norm_delta and it>2:
            break
        if delta_a.all()==0: #See svd_solve in math_func2 to check when this happens
            break

    if disp is True:
        if tol>norm_delta:
            if it<maxit:
                print('Iterations to reach convergence:',it)
            #print('Solution: a = \n',a)
        toc=time.time()
        if toc-tic<60:
            print('Elapsed time:', toc-tic, 'sec')
        else:
            print('Elapsed time:', (toc-tic)/60, 'min')
    #Save txt file
    flabel=['it','L','norm']
    for i in range(len(a)):
        flabel.append('a%g'%(i+1))
    file=np.column_stack((flabel,file))
    #filename='./txt/a_iteration_Jmax_%g_gamma2_%g.txt'%(Jmax,gamma2)
    try:
        os.mkdir('./txt/'+ffolder)
    except FileExistsError:
        print('./txt/'+ffolder+' already created')

    if isinstance(a_d, np.ndarray) is True:
        filename='./txt/'+ffolder+'a_iteration_Jmax_%g_K_%g.txt'%(Jmax,K)
    else:
        filename='./txt/'+ffolder+'a_iteration_Jmax_%g_K_%g.txt'%(Jmax,K)
    np.savetxt(filename,file,fmt='%s',delimiter='\t')
    return a



def minimization(Jmin,Jmax,a0,a_d,RHO,THETA,ap,Ok,\
disp=True,cut=None,gamma=gamma1,nuc=nuc,N=N,ffolder='',K=2):
    """
    Function that optimizes the aberration
    coefficients in the Phase Diversity process with a non linear minimization
    method
    """


    def merit_func(a):
        Hk,normhk=OTF(a,a_d,RHO,THETA,ap,norm=True,nuc=nuc,N=N,K=K) #Focused OTFs
        norma=normhk[0] #Norm for correlations

        #Q factor, merit function and focused and defocused pupils
        Q=Qfactor(Hk,gamma=gamma,nuc=nuc,N=N)
        noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N)
        Ok_filt=np.zeros((N,N,K),dtype='complex128')
        pk=np.zeros((N,N,K),dtype='complex128')
        for i in range(K):
            Ok_filt[:,:,i]=noise_filt*Ok[:,:,i] #Filtered FFT
            pk[:,:,i]=pupil(a,a_d[i],RHO,THETA,ap) #Pupil

        #Calculation of merit function
        E=meritE(Ok_filt,Hk,Q)
        e=merite(E)
        L=meritl(e,cut=cut)


        #Additive constant to the merit function (no change during optimization)
        #for i in range(K):
        #    Oshift=np.fft.fftshift(Ok[:,:,i]) #No filtering (Paxman, eq. 21)
        #    ok=N**2*np.real(np.fft.ifft2(Oshift))
        #    L-=meritl(ok,cut=cut)
        return L

    def jac_L(a):
        b=np.zeros((Jmax-1,1),dtype='float64')
        Hk,normhk=OTF(a,a_d,RHO,THETA,ap,norm=True,nuc=nuc,N=N,K=K) #Focused OTFs
        norma=normhk[0] #Norm for correlations

        #Q factor, merit function and focused and defocused pupils
        Q=Qfactor(Hk,gamma=gamma,nuc=nuc,N=N)
        noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N)
        Ok_filt=np.zeros((N,N,K),dtype='complex128')
        pk=np.zeros((N,N,K),dtype='complex128')
        for i in range(K):
            Ok_filt[:,:,i]=noise_filt*Ok[:,:,i] #Filtered FFT
            pk[:,:,i]=pupil(a,a_d[i],RHO,THETA,ap) #Pupil
        E=meritE(Ok_filt,Hk,Q)
        e=merite(E)

        #Calculation of b
        for i in range(Jmin,Jmax):
            zi=zk.zernikej_Noll(i,RHO,THETA)
            derHk_i=derHj2(zi,pk,normhk)
            #derivL[i-1]=derivmerit(Ok_filt,Hk,derHk_i,Q,gamma=gamma,cut=cut)
            derivEi=derEj(derHk_i,Ok_filt,Hk,Q,norma,gamma=gamma)
            derivei=derej(derivEi)
            b[i-1]=b_i(derivei,e,cut=cut)
        return 2*b


    meth='L-BFGS-B'
    minim=scipy.optimize.minimize(merit_func,a0,method=meth,jac=jac_L,options={'ftol':1e-11,'gtol':1e-10})
    print(minim)
    return np.array([minim.x]).T #To return an array consisting of 1 column


def read_image(file,ext,num_im=0,norma='yes',N=N):
    """
    Function that opens an image and resizes it to fill the NxN detector
    """
    if norma=='yes':
        print('Read image:',file,'is normalized by its mean !!!!!')
    if ext=='.sav':
        from scipy.io.idl import readsav
        s = readsav(file+ext)
        I = s.imagen
        I = np.array(I,dtype='float')
        N2 = I.shape[1] #Number of pixels
        if N>N2: #Repeat image to fill the detector
            I = np.pad(I,(int((N-N2)/2),int((N-N2)/2)),'wrap')
        else:
            I = I[0:N,0:N]
        return I
    elif ext=='.fits':
        from astropy.io import fits
        I=fits.open(file+ext)
        data = I[0].data

        #If first dimension corresponds to focused and defocused images
        if data.ndim==3:
            if data.shape[1]==data.shape[2]:#Reorder axes and crop the image
                data=np.moveaxis(data,0,-1)
                xmax=data.shape[0]
                xcen=int(xmax/2)
                l2=int(N/2)
                data=data[(xcen-l2):(xcen+l2),(xcen-l2):(xcen+l2),:]
        elif data.ndim==4:
            return data
        #If last dimension corresponds to focused and defocused images
        if data.shape[0]==data.shape[1]:
            if data.ndim==3: #If 'data' contains 3 dimensions
                if num_im==0: #If num_im=0, then get all images
                    data = data[:,:,:]
                else: #Else, get only the image at index num_im
                    data = data[:,:,num_im]
            elif data.ndim==2: #If 'data' containes 2 dimensions
                data=data[:,:]
            if norma=='yes': #Each image is normalized to its mean
                if data.ndim==3:
                    data = np.array(data,dtype='float64')
                    nima=data.shape[2]
                    for i in range(nima):
                        datai=data[:,:,i]
                        data[:,:,i]=datai/np.mean(datai[:,:])

                    data=data[0:data.shape[0],0:data.shape[1],:]

                    try:
                        defoc=I[1].data
                        return data,defoc
                    except:
                        return data
                elif data.ndim==2:
                    data=data/np.mean(data)
                    data=data[0:data.shape[0],0:data.shape[1]]
                    data = np.array(data,dtype='float64')
                    return data
            else:
                I = data
                I = np.array(I,dtype='float64')
                return I
        #If first dimension corresponds to focused and defocused images
        elif data.shape[0]==2:
            data1=data[0,:,:]
            I1 = data1/np.mean(data1[:])
            I1 = np.array(I1,dtype='float64')
            I1=I1[0:data.shape[1],0:data.shape[2]]
            data2=data[1,:,:]
            I2 = data2/np.mean(data2[:])
            I2 = np.array(I2,dtype='float64')
            I2=I2[0:data.shape[1],0:data.shape[2]]
            return I1, I2
        #If focused and defocused images are contained in a single image
        elif data.shape[1]==2*data.shape[0]:
            data1=data[:,:data.shape[0]]

            I1 = data1/np.mean(data1[:])
            I1 = np.array(I1,dtype='float64')
            data2=data[:,data.shape[0]:]
            I2 = data2/np.mean(data2[:])
            I2 = np.array(I2,dtype='float64')
            return I1, I2

    elif ext=='.png':
        I=im.open(file+ext)
        #I=I.convert('L')
        I=np.array(I,dtype='float')
        I=I[:N,:N,-1]
        if norma=='yes':
            I = I/np.mean(I)
        return I
    else:
        I=im.open(file+ext)
        I=I.convert('L') #Converts to black and white (not valid for np arrays)
        I=I.resize((N,N)) #Number of pixels must match detector dimensions
        I=np.array(I,dtype='float') #Converts to numpy array for operation
                                    # purposes
        return I

def save_image(ima,exitfile,folder='txt'):
    """
    This function save a numpy array with shape (N,N)
    into a FITS file
    """
    from astropy.io import fits
    hdu = fits.PrimaryHDU(ima)
    hdul=fits.HDUList([hdu])
    hdul.writeto('./'+folder+'/'+exitfile+'.fits')
    return

def gaussian_noise(SNR,image):
    row,col = image.shape
    mu  = np.mean(image)
    sigma= mu*SNR
    gauss = np.random.normal(0,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

def apod(nx,ny,perc):
   """
   Apodization window of size nx x ny. The input parameter
   perc accounts for the percentage of the window that is apodized
   """
   nx = np.int(nx)
   ny = np.int(ny)

   wx = np.ones(nx)
   wy = np.ones(ny)

   nxw = np.int(perc*nx/100.)
   nyw = np.int(perc*ny/100.)

   wi_x = 0.5*(1.-np.cos(np.pi*np.arange(0,nxw,1)/nxw))
   wi_y = 0.5*(1.-np.cos(np.pi*np.arange(0,nxw,1)/nxw))
   wx[0:nxw] = wi_x
   wx[nx-nxw:nx]= wi_x[::-1]
   wy[0:nyw] = wi_y
   wy[ny-nyw:ny]= wi_x[::-1]

   win = np.outer(wx,wy)
   return win



def scanning(data,Lsiz=128,cut=29):
    """
    This function returns an array with the subpatches of the focused
    and defocused images
    """
    #Lsiz=int(N) #Subframe size
    lsiz=int(Lsiz-2*cut) #Size of the subframe that is not overlapped to others
    i_max=int(np.floor(data.shape[0]/lsiz)-1)
    i_vec=np.arange(0,i_max)
    #Overlapping of subframes
    kk=-1
    Nima=data.shape[2]
    data_subpatch=np.zeros((int(i_max**2),Lsiz,Lsiz,Nima))
    for i in i_vec:
        xc=int(Lsiz/2+lsiz*i)
        x0=int(xc-Lsiz/2)
        xf=int(xc+Lsiz/2)
        for j in i_vec:
            kk+=1
            yc=int(Lsiz/2+lsiz*j)
            y0=int(yc-Lsiz/2)
            yf=int(yc+Lsiz/2)
            if x0<0 or y0<0:
                print('Error in scanning.py: x0 or y0 cannot be negative')
                quit()
            elif xf>data.shape[0] or yf>data.shape[1]:
                print('Error in scanning.py: xf or yf larger than of size')
                print('xf:',xf)
                print('yf:',yf)
            for n in range(Nima):
                data_subpatch[kk,:,:,n]=data[x0:xf,y0:yf,n]
    return data_subpatch

def prepare_PD(ima,nuc=nuc,N=N,wind=True,kappa=100):
    """
    This function calculates gamma for each subpatch, apodizes the subpatch
    and calculates the Fourier transform of the focused and defocused images
    """
    #Initialization of arrays
    Nima=ima.shape[2]
    gamma=np.zeros(Nima)
    gamma[0]=1
    Ok=np.zeros((ima.shape[0],ima.shape[1],Nima),dtype='complex128')

    #Calculation of gamma before apodizing
    Of=mf.fourier2(ima[:,:,0])
    for i in range(1,Nima):
        #Fourier transforms of the images and of the PSFs
        Ok[:,:,i]=mf.fourier2(ima[:,:,i])
        gamma[i]=noise_power(Of,nuc=nuc,N=N)/noise_power(Ok[:,:,i],nuc=nuc,N=N)


    #Normalization to get mean 0 and apodization
    if wind==True:
        wind=apod(ima.shape[0],ima.shape[1],10) #Apodization of subframes
    else:
        wind=np.ones((ima.shape[0],ima.shape[1]))

    #Apodization and FFT of focused image
    susf=np.sum(wind*ima[:,:,0])/np.sum(wind)
    of=(ima[:,:,0]-susf)*wind
    #Of=mf.fourier2(of)
    Of=fft2(of)
    Of=Of/(N**2)

    #Apodization and FFTs of each of the K images
    for i in range(1,Nima):
        susi=np.sum(wind*ima[:,:,i])/np.sum(wind)
        imak=(ima[:,:,i]-susi)*wind

        #Fourier transforms of the images and of the PSFs
        #Ok[:,:,i]=mf.fourier2(imak)
        Ok[:,:,i]=fft2(imak)
        Ok[:,:,i]=Ok[:,:,i]/(N**2)

        #Compute and correct the shifts between images
        error,row_shift,col_shift,Gshift=sf.dftreg(Of,Ok[:,:,i],kappa)
        Ok[:,:,i]=Gshift #We shift Ok[:,:,i]
        #Shift to center the FTTs
        Ok[:,:,i]=fftshift(Ok[:,:,i])
    Ok[:,:,0]=fftshift(Of)
    return Ok, gamma, wind, susf

def prepare_PD2(ima,nuc=nuc,N=N,wind=True):
    """
    This function apodizes each subpatch, calculates gamma AFTER apodization
    and calculates the Fourier transform of the focused and defocused images
    """
    #Normalization to get mean 0 and apodization
    if wind==True:
        wind=apod(ima.shape[0],ima.shape[1],10) #Apodization of subframes
    else:
        wind=np.ones((ima.shape[0],ima.shape[1]))

    #Focused image
    susf=np.sum(wind*ima[:,:,0])/np.sum(wind)
    of=(ima[:,:,0]-susf)*wind
    Of=mf.fourier2(of)
    Of=Of/(N**2)

    #FFTs and gamma factors of each of the K images
    Nima=ima.shape[2]
    gamma=np.zeros(Nima)
    Ok=np.zeros((ima.shape[0],ima.shape[1],Nima),dtype='complex128')

    gamma[0]=1
    Ok[:,:,0]=Of
    for i in range(1,Nima):
        susi=np.sum(wind*ima[:,:,i])/np.sum(wind)
        imak=(ima[:,:,i]-susi)*wind

        #Fourier transforms of the images and of the PSFs
        Ok[:,:,i]=mf.fourier2(imak)
        Ok[:,:,i]=Ok[:,:,i]/(N**2)

        #We calculate gamma after apodizing the images
        gamma[i]=noise_power(Of,nuc=nuc,N=N)/noise_power(Ok[:,:,i],nuc=nuc,N=N)
    return Ok, gamma, wind, susf

def object_estimate(ima,a,a_d,wind=True,cobs=0,cut=29,low_f=0.2):
    """
    This function restores the image once the Zernike coefficients are
    retrieved
    """
    #Pupil sampling according to image size
    N=ima.shape[0]
    nuc=1/(wvl*fnum) #Critical frequency (m^(-1))
    inc_nu=1/(N*Delta_x)
    R=(1/2)*nuc/inc_nu #Pupil radius [pixel]
    nuc=2*R #critical frequency (pixels)
    nuc-=1
    ap=aperture(N,R,cobs=cobs)
    RHO,THETA=sampling2(N=N,R=R)

    #Fourier transform images
    Ok, gamma, wind, susf=prepare_PD(ima,nuc=nuc,N=N,wind=wind)

    if a_d[0]==a_d[1]:#In this case, the 2nd image is a dummy image
        if a_d[0]==0:
            gamma=[1,0] #To account only for the 1st image


    #OTFs
    Hk,normhk=OTF(a,a_d,RHO,THETA,ap,norm=True,nuc=nuc,K=Ok.shape[2])

    #Restoration
    Q=Qfactor(Hk,gamma=gamma,nuc=nuc,N=N)
    noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N,low_f=low_f)

    if gamma[1]==0 and N==1536: #For FDT full images
        noise_filt=noise_filt*cir_aperture(R=nuc-200,N=N,ct=0)


    Nima=Ok.shape[2]
    for i in range(0,Nima):
        #Fourier transforms of the images and of the PSFs
        Ok[:,:,i]=noise_filt*Ok[:,:,i]

    #Compute and print merit function
    E=meritE(Ok,Hk,Q)
    e=merite(E)

    L=meritl(e,cut=cut)/(Nima*(Nima-1)/2)
    print('Optimized merit function (K=%g):'%Nima,L)


    #Restoration
    O=Ffactor(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N)

    Oshift=np.fft.fftshift(O)
    o=np.fft.ifft2(Oshift)
    #o=np.fft.fftshift(o)
    o_restored=N**2*o.real
    object=o_restored+susf

    return object,susf,noise_filt


def import_aberr(file='IMaX_aberr.txt',Jmax_ab=33):
    aberr_file=file
    IMaX_aberr=np.loadtxt(aberr_file)
    a_aberr=IMaX_aberr[:(Jmax_ab-1),1] #Aberrations induced in the image [rads]
    norm_aberr=np.linalg.norm(a_aberr)
    return a_aberr,norm_aberr

def input_params(input):
    return Jmax,dmin,dmax,deltad,NSR,w_cut,a6,Jmax_ab,width

def zernike_from_txt(path):
        filename=path
        #Import txt files
        data=np.genfromtxt(filename,delimiter='\t',unpack=False,dtype=None,\
        encoding='utf-8')
        names=np.array(data[:,0],dtype='str')
        values=data[:,1]

        #Obtain values from imported data
        a1_ind=np.argwhere(names=='a1')[0][0]
        a=np.array(values[a1_ind:],dtype='float64')
        return a

def MTF_ideal(x):
    MTF=2/np.pi*(np.arccos(x)-x*np.sqrt(1-x**2))
    return MTF

def inta_step(Nima):
    #RMS defocus step in rads
    if Nima==21:
        rad_step=0.2
    elif Nima==11:
        rad_step=0.4
    wvl_step=rad_step*np.sqrt(3)/np.pi #PV defocus step (lambda)
    return wvl_step

def open_dark(path,header):
    I=fits.open(path)
    dark=I[0].data
    dark_header=I[0].header
    if dark_header['IMGDIRX'] == 'YES':
        printc("Flipping the dark: ", color=bcolors.FAIL)
        dark = np.fliplr(dark)

    PXBEG1 = int(header['PXBEG1']) - 1
    PXEND1 = int(header['PXEND1']) - 1
    PXBEG2 = int(header['PXBEG2']) - 1
    PXEND2 = int(header['PXEND2']) - 1
    # CHECK NACC
    acc = int(header['ACCACCUM']) * int(header['ACCCOLIT'])
    acc_dark = int(dark_header['ACCACCUM']) * int(dark_header['ACCCOLIT'])
    if acc != acc_dark:
        printc('WARNING - NACC NOT IDENTICAL DURING DARK CORRECTION', color=bcolors.FAIL)
        printc('DARK NACC ', acc_dark, ' DATA NACC ', acc, color=bcolors.FAIL)

    dark = dark[PXBEG2:PXEND2 + 1, PXBEG1:PXEND1 + 1]
    return dark

def open_flat(path,header):
    I=fits.open(path)
    flat=I[0].data


    PXBEG1 = int(header['PXBEG1']) - 1
    PXEND1 = int(header['PXEND1']) - 1
    PXBEG2 = int(header['PXBEG2']) - 1
    PXEND2 = int(header['PXEND2']) - 1

    if flat.ndim==3:
        flat=flat[0,:,:] #Flat at continuum

    flat = flat[PXBEG2:PXEND2 + 1, PXBEG1:PXEND1 + 1]
    return flat
