import document_level.main
import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sift_step_size = 150
sift_cell_size = 15
sift_n_classes = 5

patch_width = 300
patch_height = 75
patch_hop_size = 25

visualize_progress=True 

searchfile = '../../george_washington_files/2700270.png'
image = Image.open(searchfile)
# Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen. 
im_arr = np.asarray(image, dtype='float32')
dimensions = im_arr.shape

# 1043 671 1443 765 companies
groundtrouth = (1043, 671, 1443, 765, "companies")
distance_to_end_x = dimensions[0]- groundtrouth[2]
distance_to_end_y = dimensions[1]- groundtrouth[3]
query = (groundtrouth[0] - (patch_width - min(distance_to_end_x, patch_width)),
         groundtrouth[1] - (patch_height - min(distance_to_end_y, patch_height)), 
         groundtrouth[0] + min(distance_to_end_x, patch_width),
         groundtrouth[1] + min(distance_to_end_y, patch_height)
         )

query_im = im_arr[query[1]:query[3], query[0]:query[2]]
if visualize_progress:
    plt.imshow(query_im, cmap=cm.get_cmap('Greys_r'))
    plt.show()

siftcalc = document_level.main.SiftCalculator(sift_step_size, sift_cell_size, sift_n_classes)
centroids, labels = siftcalc.calculate_visual_words_for_document(searchfile, visualize = visualize_progress)


print labels




if __name__ == '__main__':
    pass