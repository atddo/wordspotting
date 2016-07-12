import document_level.main
import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import patch_level.main
import latent_semantic_indexing.TopicFeatureTransform
from latent_semantic_indexing.TopicFeatureTransform import TopicFeatureTransform
from compiler.pyassem import FLAT
from tables.table import Column
from scipy.spatial.distance import cdist

sift_step_size = 30
sift_cell_size = 15
sift_n_classes = 50

patch_width = 300
patch_height =300
patch_hop_size = 100
metric = 'cosine'

flatten_dimensions = 5

visualize_progress=False 

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
query_sift = siftcalc.calculate_visual_words_for_query(query_im, visualize=visualize_progress)

print labels
fvd = patch_level.main.feature_vector_descriptor(patch_width, patch_height, patch_hop_size, patch_hop_size,sift_step_size, sift_cell_size, sift_n_classes)
patch_mat = fvd.patch_mat(dimensions[1], dimensions[0], labels, sift_cell_size, sift_step_size)

pyramid_mat = np.zeros_like(patch_mat)
for column in range(pyramid_mat.shape[1]):
    for row in range(pyramid_mat.shape[0]):
        pyramid_mat[row,column] = fvd.spatial_pyramid(patch_mat[row,column])
        
query_pyramid = fvd.spatial_pyramid(query_sift)

tft = TopicFeatureTransform(flatten_dimensions)
flat_pyramid_mat=np.array([v for i in pyramid_mat for v in i])

tft.estimate(flat_pyramid_mat)
mat_shape = pyramid_mat.shape

transfomed_array=tft.transform(flat_pyramid_mat)
transformed_query = tft.transform(np.array(query_pyramid))

distances_array = cdist(transfomed_array, np.array([transformed_query]), metric=metric)

distances_mat = distances_array.reshape(mat_shape)






if __name__ == '__main__':
    pass