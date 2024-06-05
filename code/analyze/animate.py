import matplotlib.pyplot as plt
from matplotlib import animation as ani
import numpy as np 
from IPython.display import HTML

def init_image(data,fig,ax,extent,x,y,u,v,vlim,
               subtitle,clabel,origin="lower"):
        
    img = ax.imshow(data, extent=extent,
                 origin=origin, vmin=vlim[0], vmax=vlim[1],
                 filternorm=False, animated=True, cmap="bone_r")
    if u is not None and v is not None:
        qu = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, alpha=0.5)
    else:
        qu = None
    ax.set_title(subtitle, {'fontsize': 8})     
    
        
    #gl = ax.gridlines(draw_labels=True)
    #gl.xlabels_top = False
    #gl.ylabels_right = False

    return ax,img,qu

def create_animation(data,vectors=None, extent=None,
                     out_name="out.gif", title="", origin="lower",
                     clabel="COT", vlim=[-1, 7], data2=None,
                     skip=10, subtitles = ["NN estimate","Observations"]):
    """
    Create an animation.

    Parameters
    ----------
    data : numpy.array
        Array containing cloud optical thickness data
    vectors : numpy.array
        Array of motions (wind)
    out_name : str
        Name of the output file with a proper file format (e.g .gif)
    title : str
        Title for the animation
    clabel : str
        Color bar label.
    vlim : list, optional
        Plot limits. The default is [0, 100]
    """
    if vectors is not None:
        grid_x, grid_y = np.meshgrid(np.linspace(0, vectors.shape[-2] - 1, vectors.shape[-2]),
                                 np.linspace(0, vectors.shape[-1] - 1, vectors.shape[-1]))
        x = grid_x[::skip, ::skip]
        y = grid_y[::skip, ::skip]
        u = vectors[:,0][:,::skip,::skip]
        v = vectors[:,1][:,::skip,::skip]
    else:
        x, y, u, v = None, None, None, None
    
    if data2 is None:
        fig, ax1 = plt.subplots(1,1, figsize=(10,10))
        ax1, img1, qu1 = init_image(data[0], fig, ax1, extent, x, y,
                                    u[0] if u is not None else None,
                                    v[0] if v is not None else None,
                                    vlim, subtitles[0], clabel, origin)
        ax2 = None
        clab = fig.colorbar(img1, ax=[ax1],
                            fraction=0.05,orientation="horizontal")
    else:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,10))
        ax1, img1, qu1 = init_image(data[0], fig, ax1, extent, x, y, 
                                    u[0] if u is not None else None,
                                    v[0] if v is not None else None,
                                    vlim, subtitles[0], clabel, origin)
        ax2, img2, qu2 = init_image(data2[0], fig, ax2, extent, x, y, 
                                    u[0] if u is not None else None,
                                    v[0] if v is not None else None,
                                    vlim, subtitles[1], clabel, origin)

        clab = fig.colorbar(img1, ax=[ax1,ax2],
                            fraction=0.05,orientation="horizontal")
    clab.ax.set_title(clabel)    
    t = plt.suptitle(title+", step 0", ha="center", fontsize=10, y=0.9)
    plt.subplots_adjust(hspace=0.45,bottom=0.3,top=0.8,right=0.98)
    #plt.show()
    
    def updatefig(n):
        img1.set_array(data[n, :, :])
        t.set_text(title + ", step " + str(n))

        if u is not None and v is not None:
            qu1.set_UVC(u[n, :, :], v[n, :, :])

        if np.any(data2 is None):
            if u is not None and v is not None:
                return img1, qu1, plt
            else:
                return img1, plt

        img2.set_array(data2[n, :, :])

        if u is not None and v is not None:
            qu2.set_UVC(u[n, :, :], v[n, :, :])

        if u is not None and v is not None:
            return img1, qu1, img2, qu2, plt
        else:
            return img1, img2, plt
    plt.close()
    n = data.shape[0]
    animation = ani.FuncAnimation(fig, updatefig, init_func=None,
                                  frames=n, blit=False, interval=600, repeat=True)
    animation.save(out_name, fps=1.5, dpi=400)
    return HTML(animation.to_jshtml())