"""
This program corrects raw images from dark and flat and saves it 
in a FITS file.
"""
import numpy as np
import os, sys
sys.path.append('./functions')
import pd_functions_v21 as pdf
import math_func2 as mf
import shift_func as sf
import read_functions as rf
from tqdm import tqdm
import general_func as gf
from matplotlib import pyplot as plt
from scipy import ndimage

"""
Input parameters
"""
N0=600 #Number of pixels at each direction
darkfactor=1 #1
flatshiftx=0# Shift of the flat in X direction
flatshifty=0# Shift of the flat in Y direction
bestfocusmethod='contrast' #'contrast' or 'laplacian' (to determine best focus)
ffolder='../../Colaboración INTA/2025/medidas_21_02_2025'
fname=['pdesenfocada2']
fdark=['dark_plus']
fflat=['flat_plus']
foc_index=0 #2 #Index of the focused image
namefits='mdefocus_Z4=0'#'refocfine' #Name of the saved file
ext='.fits'
xmin=0 #Minimum X coordinate when saving the image
xmax=xmin+N0 #Maximum X coordinate when saving the image
ymin=0
ymax=ymin+N0
x0c=150 #Initial X coordinate to calculate contrast
xfc=-150 #Final X coordinate to calculate contrast
y0c=x0c 
yfc=xfc 

#Name of the saved FITs file with the aligned focused and defocused image
K=len(fname)+1 #Number of images with different defocuses



"""
Search for the images, normalization and correction of dark and flat
"""

ok=pdf.read_image(ffolder+'/'+fname[0],ext=ext,N=1024,norma='no')


#Check if any pixel is saturated
ok_sat=np.where(ok==1,1,0)
sat_pix=len(np.argwhere(ok_sat>0))
if sat_pix>0:
    print('# of saturated pixels:',sat_pix)
    plt.imshow(ok_sat)
    plt.title('Saturated pixels map')
    plt.show()
    print('Program aborted')
    #quit()


fig,axs=plt.subplots()
ima=axs.imshow(ok[x0c:xfc,y0c:yfc],cmap='gray')
plt.title('Uncorrected image')
fig.colorbar(ima)
plt.show()
plt.close() 

if len(fdark)>0:
    dark=pdf.read_image(ffolder+'/'+fdark[0],ext=ext,N=1024,norma='no')

    dark_sat=np.where(dark==1,1,0)
    if len(np.argwhere(dark_sat>0))>0:
        dark=np.where(dark==1,0,dark) #Where dark is saturated, change the pixel to 0.
        print('# of pixels where dark is saturated:',len(np.argwhere(dark_sat>0)))


    if ok.ndim>2:
        for i in range(ok.shape[2]):
            ok[:,:,i]=ok[:,:,i]-dark    
    else: 
        ok=ok-dark   
        fig,axs=plt.subplots()
        ima=axs.imshow(dark[x0c:xfc,y0c:yfc],cmap='gray',vmax=np.mean(dark)+3*np.std(dark))
        plt.title('Dark')
        fig.colorbar(ima)
        plt.show()
        plt.close()
    if len(fflat)>0:
        flat=pdf.read_image(ffolder+'/'+fflat[0],ext=ext,N=1024,norma='no')
        flat=flat-dark
        flat=np.where(flat<0.001,np.mean(flat),flat)
        if ok.ndim>2:
            for i in range(ok.shape[2]):
                ok[:,:,i]=ok[:,:,i]/flat   
                ok[:,:,i]=np.where(ok[:,:,i]>0.99,np.mean(ok[:,:,i]),ok[:,:,i])
                ok[:,:,i]=np.where(ok[:,:,i]<0,0,ok[:,:,i])  
        else:    
            ok=ok/flat 
            ok=np.where(ok>0.99,np.mean(ok),ok)
            ok=np.where(ok<0,0,ok) 

            fig,axs=plt.subplots()
            ima=axs.imshow(flat[x0c:xfc,y0c:yfc],cmap='gray')
            plt.title('Flat')
            fig.colorbar(ima)
            plt.show()
            plt.close()




fig,axs=plt.subplots()
if ok.ndim>2:
    ima=axs.imshow(ok[x0c:xfc,y0c:yfc,foc_index],cmap='gray')
else:
    ima=axs.imshow(ok[x0c:xfc,y0c:yfc],cmap='gray')    
plt.title('Corrected image')
fig.colorbar(ima)
plt.show()
plt.close()



if bestfocusmethod=='laplacian':
    laplacian=ndimage.laplace(ok[x0c:xfc,y0c:yfc])
    laplacian2=np.sqrt(laplacian**2)#ndimage.laplace(laplacian)
    #Average contrast of laplacian at different parts of the limb
    contrast_lapl=np.mean(laplacian2[x0c:xfc,y0c:yfc,:],axis=(0,1))
    contrast_lapl2=np.mean(laplacian2[x0c:xfc,y0c:yfc,:],axis=(0,1))
    contrast_lapl3=np.mean(laplacian2[x0c:xfc,y0c:yfc,:],axis=(0,1))
    contrast_lapl4=np.mean(laplacian2[x0c:xfc,y0c:yfc,:],axis=(0,1))
    contrast_mean=contrast_lapl+contrast_lapl2+contrast_lapl3+contrast_lapl4

    plt.plot(contrast_mean/np.max(contrast_mean),'k')
    plt.ylabel('Average Laplacian at limb')
    plt.xlabel('Image index')
    plt.show()
    plt.close()
elif bestfocusmethod=='contrast':
    if ok.ndim>2:
        contrast=np.std(ok[x0c:xfc,y0c:yfc,:],axis=(0,1))/\
        np.mean(ok[x0c:xfc,y0c:yfc,:],axis=(0,1))
        plt.plot(contrast,marker='o')
        plt.show()
        plt.close()
    else:
        contrast=np.std(ok[x0c:xfc,y0c:yfc],axis=(0,1))/\
        np.mean(ok[x0c:xfc,y0c:yfc],axis=(0,1))
        print('Contrast (%):',np.round(contrast*100,2))






plt.close()
plt.imshow(ok[xmin:xmax,ymin:ymax],cmap='gray')
plt.show()
gf.save_fits(ok[xmin:xmax,ymin:ymax],namefits+'.fits')
