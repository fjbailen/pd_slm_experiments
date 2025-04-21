from matplotlib import pyplot as plt
import pd_functions_v21 as pdf
import numpy as np

def logplot2(I,sing=True,color=None,low=None,high=None,fourier=True):
    if fourier is True:
        N=I.shape[0]
        inc_nu=1/(N*pdf.Delta_x)
        max_nu=(N-1)*inc_nu
        nuc=pdf.nuc
        extent=[-0.5*max_nu/nuc,0.5*max_nu/nuc,0.5*max_nu/nuc,-0.5*max_nu/nuc]
        plt.xlabel('$u/\\nu_c$')
        plt.ylabel('$v/\\nu_c$')
    else:
        extent=None
    if sing is True:
        plt.imshow(np.log10(np.abs(I)**2+1),cmap=color,vmin=low,vmax=high,\
        extent=extent)
    elif sing is False:
        plt.imshow(np.log10(np.abs(I)**2),cmap=color,vmin=low,vmax=high,\
        extent=extent)
    plt.colorbar()
    plt.show()
    plt.close()

def plot2(I,color=None,low=None,high=None):
    plt.imshow(I,cmap=color,vmin=low,vmax=high)
    plt.colorbar()
    plt.show()
    plt.close()

def plot_otf(otf):
    N=otf.shape[0]
    otfrad=otf[int(N/2),:]
    inc_nu=1/(N*pdf.Delta_x)
    plt.plot((np.arange(N)-N/2)*inc_nu/pdf.nuc,np.abs(otfrad))
    plt.xlabel('$\\nu/\\nu_c$')
    plt.show()
    plt.close()

def plot2_otf(otf):
    N=otf.shape[0]
    inc_nu=1/(N*pdf.Delta_x)
    max_nu=(N-1)*inc_nu
    nuc=pdf.nuc
    plt.imshow(np.abs(otf),\
    extent=[-0.5*max_nu/nuc,0.5*max_nu/nuc,0.5*max_nu/nuc,-0.5*max_nu/nuc])
    plt.xlabel('$u/\\nu_c$')
    plt.ylabel('$v/\\nu_c$')
    plt.colorbar()
    plt.show()
    plt.close()
