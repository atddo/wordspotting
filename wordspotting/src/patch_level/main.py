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
    def __init__(self, x_size, y_size, x_step_size, y_step_size, n_classes):
        self.__x_size = x_size
        self.__y_size = y_size
        self.__x_step_size = x_step_size
        self.__y_step_size = y_step_size
        self.__n_classes = n_classes
        
    # gets the patch (row, column)
    # im_arr: the picture
    # return im_arr: sliced array im_arr
    def get_patch(self, im_arr, row, column):
        row_patch = row * self.__step_size
        column_patch = column * self.__step_size

        return im_arr[row_patch:row_patch+self.__x_size][column_patch:column_patch+self.__y_size]

    def patch_mat(self,pic_x_size, pic_y_size, picture_sift_mat, sift_hop):

        max_patches_x = (pic_x_size - self.__x_size)/self.__x_step_size +1
        max_patches_y = (pic_y_size - self.__y_size)/self.__y_step_size +1
        print max_patches_x
        print max_patches_y
        patch_mat = []

        for column in range(max_patches_y):
            for row in range(max_patches_x):


                y1 = (column*self.__y_step_size)/sift_hop
                y2 = (column*self.__y_step_size+self.__y_size)/sift_hop
                x1 = (row*self.__x_step_size)/sift_hop
                x2 = (row*self.__x_step_size+self.__x_size)/sift_hop
                #print "x1 = %d x2 = %d y1 = %d y2 = %d" %(x1, x2, y1, y2)
                patch_mat.append(picture_sift_mat[y1:y2+1,x1:x2+1])

        self.__shape=(max_patches_y,max_patches_x)
        return np.array(patch_mat).reshape(max_patches_y,max_patches_x)

    def getShape(self):
        return self.__shape
    def spatial_pyramid(self, patch_mat):

        (patch_x_size, patch_y_size) = patch_mat.shape
        total = patch_mat



        left_half = total[:,:patch_y_size/2]
        right_half = total[:,patch_y_size/2:]


        return np.vstack((self.getBagOfFeatures(total),self.getBagOfFeatures(left_half),self.getBagOfFeatures(right_half))).reshape(-1)

    def getBagOfFeatures(self, sift_mat):
        return np.bincount(sift_mat.reshape(-1),minlength = self.__n_classes)

    def detect_collision(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if x1 <= x3 & x2 > x3:
            if y1 <= y3 & y2 > y3:
                return True
            elif y1 > y3 & y1 <= y4:
                return True
        elif x1 > x3 & x1 <= x4:
            if y1 <= y3 & y2 > y3:
                return True
            elif y1 > y3 & y1 <= y4:
                return True
        else:
            return False

    def calculate_overlap(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if x1 <= x3 & x2 > x3:
            if x2 <= x4:
                x = x2 - x3
            else:
                x = x4 - x3
            if y1 <= y3 & y2 > y3:
                if y2 <= y4:
                    y = y2 - y3
                else:
                    y = y4 - y3
            elif y1 > y3 & y1 <= y4:
                if y2 <= y4:
                    y = y2 - y1
                else:
                    y = y4 - y1
        elif x1 > x3 & x1 <= x4:
            if x2 <= x4:
                x = x2 - x1
            else:
                x = x4 - x1
            if y1 <= y3 & y2 > y3:
                if y2 <= y4:
                    y = y2 - y3
                else:
                    y = y4 - y3
            elif y1 > y3 & y1 <= y4:
                if y2 <= y4:
                    y = y2 - y1
                else:
                    y = y4 - y1
        section = x * y

        r1_area = (x2 - x1) * (y2 - y1)
        r2_area = (x4 - x3) * (y4 - y3)

        union = r1_area + r2_area - section

        overlap = section / union

        return overlap