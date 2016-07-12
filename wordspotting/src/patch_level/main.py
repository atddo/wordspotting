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
    def __init__(self, x_size, y_size, x_step_size, y_step_size, sift_step_size, sift_cell_size):
        self.__x_size = x_size
        self.__y_size = y_size
        self.__x_step_size = x_step_size
        self.__y_step_size = y_step_size
        self.__sift_step_size = sift_step_size
        self.__sift_cell_size = sift_cell_size
        self.__x_sift_per_patch = np.floor((self.__x_size + 1.5 * self.__sift_cell_size) / self.__sift_step_size)
        self.__y_sift_per_patch = np.floor((self.__y_size + 1.5 * self.__sift_cell_size) / self.__sift_step_size)
    
    # gets the patch (row, column)
    # im_arr: the picture
    # return im_arr: sliced array im_arr
    def get_patch(self, im_arr, row, column):
        row_patch = row * self.__step_size
        column_patch = column * self.__step_size
        
        return im_arr[row_patch:row_patch+self.__x_size][column_patch:column_patch+self.__y_size]
        
    def patch_mat(self,pic_x_size, pic_y_size, picture_sift_mat, cell_size, sift_hop):
          
        print pic_x_size
        max_patches_x = np.floor((pic_x_size - self.__x_size)/self.__x_step_size)
        max_patches_y = np.floor((pic_y_size - self.__y_size)/self.__y_step_size)
        print max_patches_x
        patch_mat = []
        
            
        for column in range(int(max_patches_x)):
            for row in range(int(max_patches_y)):
                y1 = (column*self.__y_step_size)/sift_hop
                y2 = (column*self.__y_step_size+self.__y_size)/sift_hop
                x1 = (row*self.__x_step_size)/sift_hop
                x2 = (row*self.__x_step_size+self.__x_size)/sift_hop
                #print "x1 = %d x2 = %d y1 = %d y2 = %d" %(x1, x2, y1, y2)
                patch_mat.append(picture_sift_mat[y1:y2,x1:x2])
                
        return np.array(patch_mat)
    
    def spatial_pyramid(self, patch_mat):
        (patch_x_size, patch_y_size) = patch_mat.shape
        
        pyramid_mat = np.array(patch_x_size, patch_y_size)
        
        for i,j in patch_mat.shape:
            total = patch_mat[i][j]
            left_half = total[:patch_x_size/2]
            right_half = total[patch_x_size/2:]
            
            pyramid_mat[i][j] = np.hstack((total, left_half, right_half))
            
        return pyramid_mat