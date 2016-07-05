import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
import vlfeat
from scipy.spatial.distance import cdist

class feature_vector_descriptor(object):
    
    #
    # initializes the patch size and step size
    #
    def __init__(self, x_size, y_size, x_step_size, y_step_size):
        self.__x_size = x_size
        self.__y_size = y_size
        self.__x_step_size = x_step_size
        self.__y_step_size = y_step_size
    
    # gets the patch (row, column)
    # im_arr: the picture
    # return im_arr: sliced array im_arr
    def get_patch(self, im_arr, row, column):
        row_patch = row * self.__step_size
        column_patch = column * self.__step_size
        
        return im_arr[row_patch:row_patch+self.__x_size][column_patch:column_patch+self.__y_size]
        
    def patch_mat(self, picture_sift_mat, labels_mat, cell_size, sift_hop, x_sift_per_patch, y_sift_per_patch):
        (pic_x_size, pic_y_size) = picture_sift_mat.shape
        max_patches_x = np.floor((pic_x_size - self.__x_size)/self.__x_step_size)
        max_patches_y = np.floor((pic_y_size - self.__y_size)/self.__y_step_size)
        
        patch_mat = np.array(max_patches_x, max_patches_y)
        
        for (row, column) in picture_sift_mat.shape:
            x = np.ceil(row-1,5*cell_size/sift_hop)
            y = np.ceil(column-1,5*cell_size/sift_hop)
            
            patch_mat[row][column] = picture_sift_mat[x:x + x_sift_per_patch][y:y + y_sift_per_patch]
            
        return patch_mat
    
    def spatial_pyramid(self, patch_mat):
        (patch_x_size, patch_y_size) = patch_mat.shape
        
        pyramid_mat = np.array(patch_x_size, patch_y_size)
        
        for i,j in patch_mat.shape:
            total = patch_mat[i][j]
            left_half = total[:patch_x_size/2]
            right_half = total[patch_x_size/2:]
            
            pyramid_mat[i][j] = np.hstack((total, left_half, right_half))
            
        return pyramid_mat