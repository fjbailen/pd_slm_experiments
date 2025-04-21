"""
This program aligns with pixel accuracy a series of focused
and defocused images and saves them in a FITS file.
"""
import numpy as np
import os, sys
sys.path.append('./functions')
sys.path.append('../../Programas e2e/Fran/functions') #SPGCam_lib and read_functions
import pd_functions_v21 as pdf
import math_func2 as mf
import shift_func as sf
import read_functions as rf
from tqdm import tqdm
import general_func as gf
from matplotlib import pyplot as plt
from scipy import ndimage,optimize

"""
Input parameters
"""
N0=601 #Number of pixels in each direction
realign=True #True or False (realign images with pixel accuracy)
ffolder='./fits/2025/medidas_21_02_2025'
fname=['focused_Z4=0','pdefocus_Z4=0'] 
foc_index=0 #Index of the focused image
i_min=0 #Minimum index of  series to be extracted
i_max=1 #2  #Maximum index of  series to be extracted
ext='.fits'
x0c=150 #.Initial X coordinate to calculate contrast
xfc=-x0c # Final X coordinate to calculate contrast
y0c=x0c #
yfc=-x0c #
Nfit=1 #Index interval for fitting aroung the maximum contrast


#Accuracy for the method to find the shift among images
kappa=100 #100 for HRT. Accuracy = 1/kappa (pixel units)


"""
Open the images
"""
if len(fname)>1:
    ok=np.zeros((N0,N0,i_max+1-i_min))
    for i in range(len(fname)):
        ok[:,:,i]=pdf.read_image(ffolder+'/'+fname[i],ext=ext,N=N0)
else:        
    ok=pdf.read_image(ffolder+'/'+fname[0],ext=ext,N=N0)
ok=ok[:,:,i_min:(i_max+1)]
K=ok.shape[2]

print('Contrast focused:',np.round(100*np.std(ok[:,:,0]),2))
print('Contrast defocused:',np.round(100*np.std(ok[:,:,1]),2))

fig,axs=plt.subplots(1,2)
axs[0].imshow(ok[:,:,foc_index],cmap='gray')
axs[0].set_title('Focused image')
axs[1].imshow(ok[:,:,1],cmap='gray')
axs[1].set_title('Defocused image')
plt.show()
plt.close()



"""
Check alignment
"""
F=np.fft.fft2(ok[x0c:xfc,y0c:yfc,foc_index])
row_shift=np.zeros(ok.shape[2])
col_shift=0*row_shift

for i in range(ok.shape[2]):
    print(i)
    G=np.fft.fft2(ok[x0c:xfc,y0c:yfc,i])
    error,row_shift[i],col_shift[i],Gshift=sf.dftreg(F,G,kappa)
    print('Shifts:',row_shift[i],col_shift[i])
    #error,row_shift2,col_shift2,Gshift=sf.dftreg(F,Gshift,2*kappa)
    #print(row_shift2,col_shift2)

    #Realign
    if realign is True:
        ok[:,:,i]=np.roll(ok[:,:,i], int(round(row_shift[i])), axis=0)
        ok[:,:,i]=np.roll(ok[:,:,i], int(round(col_shift[i])), axis=1)

if realign is True:
    gf.save_fits(ok[:N0,:N0,:],'realigned_'+fname[0]+'_'+fname[1]+'.fits')


fig,axs=plt.subplots(1,2)
axs[0].imshow(ok[:,:,foc_index],cmap='gray')
axs[0].set_title('Realigned focused ')
axs[1].imshow(ok[:,:,1],cmap='gray')
axs[1].set_title('Realigned defocused')
plt.show()
plt.close()


plt.plot(row_shift,linestyle='-',marker='o',label='X shift')
plt.plot(col_shift,linestyle='-',marker='o',label='Y shift')
plt.ylabel('Pixel shift')
plt.xlabel('Index')
plt.legend()
plt.show()
