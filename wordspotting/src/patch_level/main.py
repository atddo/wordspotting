import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image

class feature_vector_descriptor(object):
    
    #
    # initializes the patch size and step size
    #
    def __init__(self, x_size, y_size, step_size):
        self.__x_size = x_size
        self.__y_size = y_size
        self.__step_size = step_size
    
    # loading the page
    # this_page: string, name of the png
    # return im_arr: np.array of the png
    def load_page(self, this_page):
        image = Image.open(this_page)
        # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen. 
        im_arr = np.asarray(image, dtype='float32')
        # Die colormap legt fest wie die Intensitaetswerte interpretiert werden.
        plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
        plt.show()
        
        return im_arr
    
    def patch_mat(self):
        patch_per_x = 
        patch_per_y =
        result_mat = np.array()