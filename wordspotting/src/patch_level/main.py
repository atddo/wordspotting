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
    def __init__(self, x_size, y_size, x_step_size, y_step_size, sift_step_size, sift_cell_size, n_classes):
        self.__x_size = x_size
        self.__y_size = y_size
        self.__x_step_size = x_step_size
        self.__y_step_size = y_step_size
        self.__sift_step_size = sift_step_size
        self.__sift_cell_size = sift_cell_size
        self.__x_sift_per_patch = np.floor((self.__x_size + 1.5 * self.__sift_cell_size) / self.__sift_step_size)
        self.__y_sift_per_patch = np.floor((self.__y_size + 1.5 * self.__sift_cell_size) / self.__sift_step_size)
        self.__n_classes = n_classes
    # gets the patch (row, column)
    # im_arr: the picture
    # return im_arr: sliced array im_arr
    def get_patch(self, im_arr, row, column):
        row_patch = row * self.__step_size
        column_patch = column * self.__step_size
        
        return im_arr[row_patch:row_patch+self.__x_size][column_patch:column_patch+self.__y_size]
        
    def patch_mat(self,pic_x_size, pic_y_size, picture_sift_mat, cell_size, sift_hop):
          
        print pic_x_size
        max_patches_x = (pic_x_size - self.__x_size)/self.__x_step_size +1
        max_patches_y = (pic_y_size - self.__y_size)/self.__y_step_size +1
        print max_patches_x
        print max_patches_y
        patch_mat = []
        
        for row in range(max_patches_y):
            for column in range(max_patches_x):

                y1 = (column*self.__y_step_size)/sift_hop
                y2 = (column*self.__y_step_size+self.__y_size)/sift_hop
                x1 = (row*self.__x_step_size)/sift_hop
                x2 = (row*self.__x_step_size+self.__x_size)/sift_hop
                #print "x1 = %d x2 = %d y1 = %d y2 = %d" %(x1, x2, y1, y2)
                patch_mat.append(self.spatial_pyramid(picture_sift_mat[x1:x2,y1:y2]))
                
        return np.array(patch_mat).reshape(max_patches_x,max_patches_y)
    
    def spatial_pyramid(self, patch_mat):

        (patch_x_size, patch_y_size) = patch_mat.shape
        total = patch_mat

        
        left_half = total[:,:patch_y_size/2]
        right_half = total[:,patch_y_size/2:]


        return np.vstack((self.getBagOfFeatures(total),self.getBagOfFeatures(left_half),self.getBagOfFeatures(right_half))).reshape(-1)
    
    def getBagOfFeatures(self, sift_mat):
        return np.bincount(sift_mat.reshape(-1),minlength = self.__n_classes)
        
        
        