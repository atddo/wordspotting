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

    @staticmethod
    def detect_collision(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if x1 < x3 and x2 > x3:      # x3 liegt zwischen x1 und x2
            if y1 < y3 and y2 > y3:  # y3 liegt zwischen y1 und y2
                return True
            elif y1 > y3 and y1 < y4:# y1 liegt zwischen y3 und y4
                return True
        elif x1 > x3 and x1 < x4:    # x1 liegt zwischen x3 und x4
            if y1 < y3 and y2 > y3:  # y3 liegt zwischen y1 und y2
                return True
            elif y1 > y3 and y1 < y4:# y1 liegt zwischen y3 und y4
                return True
        return False


    @staticmethod
    def calculate_overlap(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if not feature_vector_descriptor.detect_collision(self, x1, y1, x2, y2, x3, y3, x4, y4):
            return 0.
         
        if x1 <= x3 and x2 > x3:
            if x2 <= x4:
                x = x2 - x3
            else:
                x = x4 - x3
            if y1 <= y3 and y2 > y3:
                if y2 <= y4:
                    y = y2 - y3
                else:
                    y = y4 - y3
            elif y1 > y3 and y1 <= y4:
                if y2 <= y4:
                    y = y2 - y1
                else:
                    y = y4 - y1
        elif x1 > x3 and x1 <= x4:
            if x2 <= x4:
                x = x2 - x1
            else:
                x = x4 - x1
            if y1 <= y3 and y2 > y3:
                if y2 <= y4:
                    y = y2 - y3
                else:
                    y = y4 - y3
            elif y1 > y3 and y1 <= y4:
                if y2 <= y4:
                    y = y2 - y1
                else:
                    y = y4 - y1
        section = x * y

        r1_area = max(x1-x2,x2-x1) * max(y1-y2,y2-y1)
        r2_area = max(x3-x4,x4-x3) * max(y3-y4,y4-y3)

        union = r1_area + r2_area - section

        overlap = section / union

        return overlap
    
if __name__ == "__main__":
    print "Collisiontest"
    a = (5,5,10,10)
    b = [(0,0,6,6), (6,0,7,7), (9,0,11,6), (9,6,11,7), (9,9,11,11), (7,9,8,11), (3,9,6,11), (3,6,6,7),
         (0,0,5,5), (5,0,10,5), (10,10,11,11), (0,10,5,12), (6,0,7,5)]
    c = [True] * 8 + [False] * 5
    print c
    for i in range(len(b)):
        print "\nTeste Rechtecke"
        print "a: %d %d %d %d" %a
        print "b: %d %d %d %d" %b[i]
        print "detect_collision:"
        tmp = feature_vector_descriptor.detect_collision(None, a[0], a[1], a[2], a[3], b[i][0], b[i][1], b[i][2], b[i][3])
        print tmp
        print "erwarteter Output:"
        print c[i]
        if tmp != c[i]:
            print "FEHLER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print "overlap: %g" %feature_vector_descriptor.calculate_overlap(None, a[0], a[1], a[2], a[3], b[i][0], b[i][1], b[i][2], b[i][3])
