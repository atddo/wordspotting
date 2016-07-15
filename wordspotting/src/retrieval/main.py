import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def retrieval():
    
    raise NotImplementedError('Implement me')



"""Usage:
    Gebruacht ist noch die distance_matrix!

    ret = Retrieval(patch_width, patch_height, patch_hop_size, img_path)
    _, non_max_list = ret.non_maximum_suppression(distance_matrix)
    coordinates_list = ret.create_list(non_max_list, visualize = True)
    
"""
class Retrieval(object):
    def __init__(self, patch_width, patch_height, patch_hop_size, img_path):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_hop_size = patch_hop_size
        image = Image.open(img_path)
        self.im_arr = np.asarray(image, dtype='float32')
        

    def non_maximum_suppression(self, distance_matrix):
        mask_x = int(np.ceil((self.patch_width - self.patch_hop_size) / float(self.patch_hop_size)))
        mask_y = int(np.ceil((self.patch_height - self.patch_hop_size) / float(self.patch_hop_size)))
        result = np.zeros_like(distance_matrix, dtype = float)
        result_coordinates = []
        for x in range(result.shape[1]):
            for y in range(result.shape[0]):
                minimum = np.min(distance_matrix[max(0,y-mask_y):min(distance_matrix.shape[0],y+mask_y+1),max(0,x-mask_x):min(distance_matrix.shape[1],x+mask_x+1)])
                if distance_matrix[y,x] == minimum:
                    result_coordinates.append((y,x,minimum))
                    result[y,x] = minimum
                else:
                    result[y,x] = None
        dtype = [('y', int), ('x', int), ('value', float)]            
        return result, np.array(result_coordinates, dtype)
    
    def create_list(self, non_max_list, visualize = True):
        # erwartet result_coordinates liste aus non_maximum_suppression
        non_max_list =  np.sort(non_max_list, order = 'value')
        result_list = []
        for (y,x,value) in non_max_list:
            elem = (x*self.patch_hop_size,y*self.patch_hop_size,x*self.patch_hop_size + self.patch_width,y*self.patch_hop_size + self.patch_height)
            patch = self.im_arr[elem[1]:elem[3] , elem[0]:elem[2]]
            result_list.append(elem)
            if visualize == True:
                print "Value is: %f" %value
                print elem
                try:
                    plt.imshow(patch, cmap=cm.get_cmap('Greys_r'))
                    plt.show()
                except ValueError:
                    print patch
                
            
        return result_list
            
if __name__ == '__main__':
    distance_matrix = np.array([5,0,2,1,0,3,4,2,2,1,1,0,2,3,4,1]).reshape(4,4)
    print distance_matrix
    ret = Retrieval(400, 300, 200, '../../george_washington_files/2710271.png')
    non_max_matrix, non_max_list =  ret.non_maximum_suppression(distance_matrix)
    print non_max_matrix
    ret.create_list(non_max_list)
    