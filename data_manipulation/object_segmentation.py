import numpy as np
import sys

# where is tumor in images
# result[0] - z axis, result[1]/result[2] - x/y axes, result[3] - RGB 
def occurence_in_layer(vol, color):
    """func returns array of numbers representing number of color pixels for each layer 
    occurence_in_layer(seg_ims_small, DEFAULT_TUMOR_COLOR) returns num of pixels damaged by tumor
    """
    result = np.array( np.equal( vol, color) )
    replace = result.all(axis = -1)
    layers = np.sum( replace, axis = tuple( range( 1, replace.ndim ) ) )
    return layers


#get the part of image where kidneys/tumors are 
def get_relevant_interval(layers):
    """func returns the first and last occurence of organ(kidney) or tumor in layers  """
    #rmin is index from legs to head
    #rmax is index from head to legs
    if( np.max(layers) == 0 and np.min(layers) == 0):
        #special case rmin -> larger then any of rmax (sys.maxsize)
        #special case rmax -> smaller then any of rmin (-1)
        return sys.maxsize, -1
    
    expanded = np.expand_dims(layers, axis=0)
    rows = np.any(expanded, axis=0)
    rmin, rmax= np.where(rows)[0][[0, -1]]
    
    if( layers[0] > 0):
        print("Watch out! Kideny/tumor is not in working volume as whole part. We must resize working volume.\nDecrease start index a bit when loading.")
    elif( layers[ len(layers) -1 ] > 0 ):        
        print("Watch out! Kideny/tumor is not in working volume as whole part. We must resize working volume.\nIncrease end index a bit when loading.")
    return (rmin,rmax)


def count_bounding_box(img):
    """func returns bounding box coords for image - gets first activation of pixels from left/right up/down"""
    # rmin,cmin*-----------
    #          |          |
    #          |          |
    #          -----------*rmax,cmax

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax]


def get_global_coords(coords):
    """func returns coords for the largest bounding box so everything fits there"""
    rmin_global = np.amin(coords[:,0],axis=0)
    rmax_global = np.amax(coords[:,1],axis=0)
    cmin_global = np.amin(coords[:,2],axis=0)
    cmax_global = np.amax(coords[:,3],axis=0)
    
    return (rmin_global, rmax_global, cmin_global, cmax_global)


def create_bounding_box(new_volume, rmin_global, rmax_global, cmin_global, cmax_global):
    """func draw bounding box along whole series of pictures (applied on the whole new_volume)
    """
    #global - biggest,smallest coordinates throught whole dataset
    new_volume[:, rmin_global -2 : rmin_global , :] = [0,255,0]
    new_volume[:, rmax_global : rmax_global +2, :]  = [0,255,0]
    new_volume[:, :, cmin_global -2 : cmin_global]  = [0,255,0]
    new_volume[:, :, cmax_global : cmax_global +2]  = [0,255,0]
    
    return new_volume