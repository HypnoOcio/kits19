import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed
import time
#https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
    
def show_film(slice_num, vol):
    plt.figure(2)
    plt.imshow(vol[slice_num])
    print(f'num of slice actual showing {slice_num}')
    plt.show()
    
def slice_viewer(volume, step = 5):
    """func shows the sequence of pictures from numpy array"""
    interactive_plot = interactive( show_film ,slice_num = (0, volume.shape[0] - 1, step ), vol = fixed(volume) )
    output = interactive_plot.children[-1]
    output.layout.height = '350px'
    return interactive_plot