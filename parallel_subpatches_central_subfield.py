"""
This program carries out the optimization of the Zernike coefficients
through the use of the phase diversity method.
"""
from PIL import Image as im
import numpy as np
import os
import time
import sys
sys.path.append('./functions')
import pd_functions_v21 as pdf
import math_func2 as mf
import zernike as zk
import multiprocessing as mtp
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage


#Parameters of input data
K=2 #Number of PD images in the series (2 = classic PD case)
Nima=2 # Number of images of the FITS file to be employed
foc_index=0 #Index of the best-focus image
index_pos=np.array([foc_index,1]) #Array with indices (1st=focused)
ffolder='./fits/2025/medidas_21_02_2025'
fname='realigned_focused_Z4=0_pdefocus_Z4=0' #Name of the FITS file
check_images=True #True to display defocuses and plot two images
Nsize=601 #1024 #Size of the original image
Jmin=4 #Minimum Noll's index of the zernike to be fitted (>=1)
Jmax=22 #16 or 22.  Highest Noll's index to be fitted = Jmax-1

#Region of images to be employed and SVD cut-off value
w_cut=0.02
x0c=50 #350 #Initial X coordinate
xfc=-50 #-x0c #Final X coordinate
y0c=x0c
yfc=xfc

#Zernike reconstruction settings
n_cores=1 #mtp.cpu_count(). Number of cores for subfielding. Max. 28 on my PC

#Convergence criteria and regularization parameters
tol=0.05 #0.01 #Tolerance criterium for stopping iterations (Bonet's powerpoint)
low_f=0.2 # 0.2 (nominal), 0 (min), 1 (max). Cut-off for optimum filter
maxnorm=2 #0.8 #Maximum norm of the solution at each iteration [rads]
maxit=10 #Maximum number of iterations
cut=int(0.15*pdf.N)#29#int(0.1*pdf.N) #None#Subframing crop to avoid edges
scan='yes' #Scan all the image subframe by subframe
svd_meth='svd' #'svd' or 'lstsq'

#Defocuses of the images
stp=pdf.inta_step(Nima) #Step of refocusing mechanism (in waves units)

a_d=np.arange(0,Nima*stp,stp) #Array with defocuses
a_d=a_d-a_d[foc_index] #Offset for focused image to have a 0 defocus
a_d=a_d[index_pos] #To get only the defocuses we are interested
a_d=a_d*np.pi/(np.sqrt(3)) #Defocus Z4 in radian


#Initial guess
a0=np.zeros((Jmax-1,1)) #Initial guess for PD optimization [rads]


"""
Read image
"""
ext='.fits'
data=pdf.read_image(ffolder+'/'+fname,ext=ext,N=Nsize)
data=data[x0c:xfc,y0c:yfc,index_pos]



#Check some parameters and two of the PD images
if check_images is  True:
    print('Step (PV lambda):',stp)
    print('Defocuses (PV lambda):',a_d*np.sqrt(3)/np.pi)
    print('Data shape:',data.shape)
    print('Position of images:',index_pos)

    fig,axs=plt.subplots(1,2)
    axs[0].imshow(data[:,:,0],cmap='gray')
    axs[1].imshow(data[:,:,-1],cmap='gray')
    plt.show()


    #Compute contrasts
    contrast=100*np.std(data[:,:,0],axis=(0,1))/\
    np.mean(data[:,:,0],axis=(0,1))
    print('Contrast focused (%):',np.round(contrast,2))

    contrast=100*np.std(data[:,:,1],axis=(0,1))/\
    np.mean(data[:,:,1],axis=(0,1))
    print('Contrast defocused (%):',np.round(contrast,2))
    quit()


"""
Preparation of PD
"""
RHO,THETA=pdf.sampling2() #Meshgrid with polar coordinates

ap=pdf.aperture(pdf.N,pdf.R) #Array with aperture of the telescope


if pdf.N>64:
    data_array=pdf.scanning(data,Lsiz=pdf.N,cut=cut) #To subfield into pdf.N x pdf.N subpatches
    k_vec=np.arange(0,data_array.shape[0]) #Vector with labels for each subimage

else:
    data_array=np.zeros(0)
    k_vec=np.zeros(1)


#Function to optimize
def subpatch(k):

    """
    PD
    """
    #Outputs of prepare_D
    if data_array.shape[0]==0:#If no subfielding
        Ok,gamma,wind,susf=pdf.prepare_PD(data)
    else: #If subfielding (pdf.N < data X/Y dimensions)
        Ok,gamma,wind,susf=pdf.prepare_PD(data_array[k,:,:,:])

    
    #gamma=[1,1]
    #print('WARNING: gamma set manually to 1')    

    if check_images is True:
        fig,axs=plt.subplots(1,2)
        cir_obs=pdf.pmask()
        axs[0].imshow(np.abs(Ok[:,:,0])**2*cir_obs)#,norm=LogNorm())
        axs[1].imshow(np.abs(Ok[:,:,1])**2*cir_obs)#,norm=LogNorm())

        print(np.sum(np.abs(Ok[:,:,0])**2*cir_obs))
        print(np.sum(np.abs(Ok[:,:,1])**2*cir_obs))
        plt.show()
        quit()    

    #Call to optimization function
    a=pdf.loop_opt(tol,Jmin,Jmax,w_cut,maxnorm,maxit,\
    a0,a_d,RHO,THETA,ap,Ok,cut=cut,method=svd_meth,gamma=gamma,K=K,low_f=low_f)
    norm_a=np.linalg.norm(a)

    #a=pdf.minimization(Jmin,Jmax,a0,a_d,RHO,THETA,ap,Ok,\
    #cut=cut,gamma=gamma,ffolder='',K=K)

    """
    Save txt and images
    """
    #Aberrations and parameters txt file
    flabela=['file','output','ext','Jmin','Jmax',\
    'tol','maxnorm','maxit','w_cut','d','a_d']
    for ai in range(len(a)):
        flabela.append('a%g'%(ai+1))
    param=np.array([fname,ffolder,ext,Jmin,Jmax,\
    tol,maxnorm,maxit,w_cut,a_d*np.sqrt(3)/np.pi,a_d],dtype=object)
    filea=np.concatenate((param.reshape(len(param),1),a))
    filea=np.column_stack((flabela,filea))
    filename='./txt/a_optimized_Jmax_%g_k_%g.txt'%(Jmax,k)
    np.savetxt(filename,filea,delimiter='\t',fmt='%s',encoding='utf-8')

    #Process information
    proc_name = mtp.current_process().name
    print('Process', proc_name)


#If no subfielding:
if data_array.shape[0]==0: #For FDT
    subpatch(0)
#If subfielding, parallel computing:
else:
    #Parallel computing!!!!
    if __name__ == '__main__':
        p=mtp.Pool(n_cores)
        p.map(subpatch,k_vec,chunksize=1)
