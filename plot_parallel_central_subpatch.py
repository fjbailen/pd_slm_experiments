import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
import numpy as np
import math_func2 as mf
import pd_functions_v21 as pdf
import os
import general_func as gf
from matplotlib.ticker import MultipleLocator
"""
Imports and plots the set of Zernike coefficients
"""
Nima=2 # Number of images in FITS
foc_index=0 #Index of the best-focus image
index_pos=np.array([foc_index,1]) #Indices to be employed (1st=focused)
N0=pdf.N #Size of the image
ffolder='./fits/2025/medidas_21_02_2025'
fname='realigned_focused_Z4=0_mdefocus_Z4=0'
txtfolder='2025/medidas_21_02_2025/'+fname

Jmin=4 #Minimum Noll's index of the zernike to be corrected (>=1)
Jmax=22 #16,22 #Maximum index of the zernikes
cut=int(0.15*pdf.N) #None#Subframing crop to avoid edges

#Cut of the filter for the image reconstruction
low_f=0.2#0.6  #0.2 (nominal), 0 (min), 1(max)

#Region of images to be employed (to avoid edges)
x0c=50 #350 #Initial X coordinate
xfc=-51 #-x0c #Final X coordinate
y0c=x0c
yfc=xfc

#Defocus
stp=pdf.inta_step(Nima) #Step of refocusing mechanism
print('Refocusing mechanism step:',stp,'waves')
a_d=np.arange(0,Nima*stp,stp)
a_d=a_d-a_d[foc_index] #Offset for focused image to have a 0 defocus
a_d=a_d[index_pos] #To get only the defocuses we are interested
a_d=a_d*np.pi/(np.sqrt(3)) #Defocus in radians

print('PD defocuses:',a_d*np.sqrt(3)/np.pi,'waves')


datafolder='./txt/'+txtfolder +'/' #Folder where txt data is saved
output='./results/'


try:
    os.mkdir(output)
except FileExistsError:
    print(output+' already created')



#Colormap limits for wavefront representation
Npl=1
vmin=-np.pi #Typically, pi/2 or pi
vmax=np.pi



"""
Read image
"""

ext='.fits'
ima=pdf.read_image(ffolder+'/'+fname,ext=ext)
#ima=ima[:N0,:N0,:]
ima=ima[x0c:xfc,y0c:yfc,:]
ima=ima[:,:,index_pos]


#Padding to restore the whole image
pad_width = int(N0*10/(100-10*2))
ima_pad= np.zeros((N0+pad_width*2,N0+pad_width*2,ima.shape[2]))
for i in range(ima.shape[2]):
    ima_pad[:,:,i] = np.pad(ima[:,:,i], pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='symmetric')



#Plot options
merit_plot=False
zernike_plot=True
multiple=True


"""
Loop
"""



if ima.ndim==3:
    data_array=pdf.scanning(ima,Lsiz=pdf.N,cut=cut)
    k_vec=np.arange(0,data_array.shape[0])
else:
    data_array=ima
    k_vec=[]
if len(k_vec)==0:
    k_vec=np.arange(1)


k_max=k_vec.shape[0]
RHO,THETA=pdf.sampling2()
ap=pdf.aperture(pdf.N,pdf.R)


L_last=np.zeros(k_max)
rms_error=np.zeros(k_max)
norm_a_last=np.zeros(k_max)
it_v=np.zeros(k_max)
av=np.zeros((Jmax-1,k_max))
rms_labels=[]


fig,axs=plt.subplots(Npl,Npl)
fig2,axs2=plt.subplots(Npl,Npl)
fig3,axs3=plt.subplots(Npl,Npl)


print('Txt folder:',datafolder)

for k in k_vec:
    """
    Results after optimization files
    """
    filename=datafolder+'a_optimized_Jmax_%g_k_%g.txt'%(Jmax,k)
    #Import txt files
    data=np.genfromtxt(filename,delimiter='\t',unpack=False,dtype=None,\
    encoding='utf-8')
    names=np.array(data[:,0],dtype='str')
    values=data[:,1]

    #Obtain values from imported data
    a1_ind=np.argwhere(names=='a1')[0][0]
    a=np.array(values[a1_ind:],dtype='float64')
    av[:,k]=a
    norm_a=2*np.pi/np.linalg.norm(a)

    maxit_ind=np.argwhere(names=='maxit')
    maxit=float(values[maxit_ind])
    wcut_ind=np.argwhere(names=='w_cut')
    w_cut=float(values[wcut_ind])

    #Wavefront
    wavef=pdf.wavefront(a,0,RHO,THETA,ap)


    """
    Plots
    """
    #Unravel coordinates of each image
    if len(k_vec)>1:
        n_vec,m_vec=np.unravel_index(k_vec,(Npl,Npl))
        n=n_vec[k]
        m=m_vec[k]

        #Zernike coefficients, original subframes and wavefront

        min_ima=np.min(data_array[:,:,0])
        max_ima=np.max(data_array[:,:,0])
        axs[n,m].plot(range(Jmin,Jmax),a[(Jmin-1):]/(2*np.pi),marker='.',label='k%.3g'%k,color='k')
        axs2[n,m].imshow(data_array[k,cut:-cut,cut:-cut,0],cmap='gray',vmin=min_ima,vmax=max_ima)
        axs3[n,m].imshow(wavef,vmin=vmin,vmax=vmax,cmap='seismic')
        axs3[n,m].set_xticks([])
        axs3[n,m].set_yticks([])
        axs3[n,m].set_title(r'$\lambda$/%.3g'%round(norm_a,2))




#plt.savefig(output + 'individual_wavefronts_Jmax_%g'%Jmax+'_'+Kname+'.png',dpi='figure')
if len(k_vec)>1:
    plt.show()
    plt.close()


"""
Average wavefront
"""
a_aver=np.mean(av,axis=1)
print('a4 (lambda):',a_aver[3]/(2*np.pi))
print('a6 (rad):',a_aver[5])
print('Defocus (mm):',np.round(np.sqrt(3)/np.pi*pdf.wvl*a_aver[5]*8*pdf.fnum**2/1e-3,2))
a_rms=np.std(av,axis=1)
norm_aver=2*np.pi/np.linalg.norm(a_aver)
print('Wavefront error:','lambda/%g'%np.round(norm_aver,1))
wavef_aver=pdf.wavefront(a_aver,0,RHO,THETA,ap)

#Average zernike coefficients and wavefront
fig4,axs4=plt.subplots()
#axs4.plot(range(Jmin,Jmax),a_aver[(Jmin-1):]/(2*np.pi),marker='o')
axs4.errorbar(range(Jmin,Jmax),a_aver[(Jmin-1):]/(2*np.pi), yerr=a_rms[(Jmin-1):]/(2*np.pi),fmt='-o',capsize=3,color='k',label='Retrieved')
axs4.set_ylabel(r'Zernike coefs. [$\lambda$]')
axs4.set_xlabel('Zernike index')
plt.grid(axis='both', which='both')


#axs4.plot(range(Jmin,Jmax),a_aberr[(Jmin-1):Jmax]/(2*np.pi),marker='o',label='Input')

try:
    axs4.plot(range(Jmin,Jmax),a_aberr[(Jmin-1):(Jmax_ab-1)]/(2*np.pi),marker='o',label='Input')
    plt.legend()
except:
    print('WARNING: Input aberrations were not plotted')

axs4.xaxis.set_minor_locator(MultipleLocator(1))
axs4.xaxis.set_major_locator(MultipleLocator(5))

#plt.savefig(output + 'zernike_plot_Jmax_%g'%Jmax+'_'+Kname+'.png',dpi='figure')


#Average wavefront aberration
fig5,axs5=plt.subplots()
axs5.imshow(wavef_aver,vmin=vmin,vmax=vmax,cmap='seismic')
axs5.set_xticks([])
axs5.set_yticks([])
axs5.set_title(r'$\lambda$/%.3g'%round(norm_aver,2))

#plt.savefig(output + 'wavefront_Jmax_%g'%(Jmax)+'_'+Kname+'.png',dpi='figure')



"""
Restoration with average Zernike coefficients
"""

wind_opt=True #True to apodize the image
print('Restoration with average Zernike coefficients')
o_plot,susf,noise_filt=pdf.object_estimate(ima_pad,a_aver,a_d,wind=wind_opt,low_f=low_f)
print('Restoration with flat wavefront')
o_plot0,susf0,noise_filt0=pdf.object_estimate(ima_pad,0*a_aver,a_d,wind=wind_opt,low_f=low_f)
x0c=cut
xfc=-cut

contrast_0=np.std(ima_pad[x0c:xfc,x0c:xfc,0])/(np.mean(ima_pad[x0c:xfc,x0c:xfc,0]))*100
contrast_rest=np.std(o_plot[x0c:xfc,x0c:xfc])/(np.mean(o_plot[x0c:xfc,x0c:xfc]))*100
min_rest=np.min(o_plot[cut:-cut,cut:-cut])
max_rest=np.max(o_plot[cut:-cut,cut:-cut])


#Original noise-filtered
ima_fourier=np.fft.fft2(ima_pad[:,:,0])
ima_fourier=np.fft.fftshift(ima_fourier)
ima_filt=ima_fourier*noise_filt
ima_filt=np.fft.fftshift(ima_filt)
ima_filt=np.fft.ifft2(ima_filt)
ima_filt=ima_filt.real



#Original and restored images
fig6,ax6=plt.subplots(1,2)
fig6.set_size_inches(7.5,4.5)
fig6.subplots_adjust(top=0.975,bottom=0.075,left=0.09,right=0.99,hspace=0.0,
wspace=0.2)
Nt=ima_pad[cut:-cut,cut:-cut,0].shape[0]
ticks=np.arange(0,Nt,80,dtype=int)
ticks=ticks[1:] #First tick at 0 position to be avoided



xd0=cut#720
xdf=-cut#1320
yd0=cut#200
ydf=-cut#800
plot1=ax6[0].imshow(ima_filt[xd0:xdf,yd0:ydf],cmap='gray',vmin=min_rest,vmax=max_rest)
plot2=ax6[1].imshow(o_plot[xd0:xdf,yd0:ydf],cmap='gray',vmin=min_rest,vmax=max_rest)

for i in range(2):
    ax6[i].set_xticks([])
    ax6[i].set_yticks([])

contrastplot2=np.round(100*np.std(o_plot[xd0:xdf,yd0:ydf])/\
np.mean(o_plot[xd0:xdf,yd0:ydf]),1)
contrastplot0=np.round(100*np.std(o_plot0[xd0:xdf,yd0:ydf])/\
np.mean(o_plot0[xd0:xdf,yd0:ydf]),1)
contrastplot1=np.round(100*np.std(ima_pad[xd0:xdf,yd0:ydf,0])/\
np.mean(ima_pad[xd0:xdf,yd0:ydf,0]),1)
#ax6[0].annotate(str(contrastplot1)+'%',(50,50),color='w',fontsize=14)
#ax6[1].annotate(str(contrastplot2)+'%',(50,50),color='w',fontsize=14)
print('Contrast focused:', contrastplot1)
print('Contrast restored:',contrastplot2)
#plt.savefig(output+'restored.eps',dpi=600)


#Only restored image
fig,ax=plt.subplots()
plot=ax.imshow(o_plot[xd0:xdf,yd0:ydf],cmap='gray',vmin=min_rest,vmax=max_rest)
#ax.annotate(str(contrastplot2)+'%',(N0-150,N0-100),color='w',fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
#ax.set_title('Restored')


#Filter
fig7,ax7=plt.subplots()
ax7.imshow(noise_filt)
#plt.savefig(output + 'noise_filter_Jmax_%g'%(Jmax)+'_'+Kname+'.png',dpi='figure')
plt.show()
plt.close()
