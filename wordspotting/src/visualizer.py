

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import Image


class ScoreVisualization(object):

    def __init__(self, score_cmap_id='jet'):
        '''
        '''
 
        self.__score_cmap_id = score_cmap_id
        
    def visualize_score_mat(self, document_name, score_mat_, score_mat_bounds, 
                            normalize_scores=True, fig=None):
        
        if score_mat_bounds is None or len(score_mat_bounds) != 4:
            raise ValueError('score_mat_bounds undefined, tuple (x_min,y_min,x_max,y_max) required')
            
        score_mat_extent = (score_mat_bounds[0], score_mat_bounds[2], 
                            score_mat_bounds[3], score_mat_bounds[1])
        
        score_mat = np.empty_like (score_mat_)
        score_mat[:] = score_mat_
        if normalize_scores:
        
            # Remove undefined HMM scores
            undef_score = np.max(score_mat) + 1
            score_mat[score_mat == -1] = undef_score
        
            # Normalize to [0,1] 
            # --> 0 best, 1 worst score
            min_score = np.min(score_mat)
            score_mat -= min_score
            max_score = np.max(score_mat)
            if max_score != 0:
                score_mat /= max_score
            # --> 1 best, 0 worst score
            score_mat -= 1
            score_mat *= -1
        
        
        #page_img_path = self.__config.get_document_image_filepath(document_name)
        #page_img = mpimg.imread(page_img_path)
        page_img = Image.open(document_name)
        page_img = np.asarray(page_img, dtype='float32')
        # page_img = np.flipud(page_img)
        
        score_mat_shape = (int(score_mat_bounds[3] - score_mat_bounds[1]), 
                           int(score_mat_bounds[2] - score_mat_bounds[0]))

        score_mat = scipy.misc.imresize(score_mat, score_mat_shape, 
                                        interp='nearest')
       
        if fig is None:
            fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.imshow(page_img, cmap=cm.get_cmap('Greys_r'))
        ax.autoscale(enable=False)
        ax.hold(True)
        ax.imshow(score_mat, cmap=cm.get_cmap(self.__score_cmap_id), 
                  alpha=0.75, extent=score_mat_extent)
        ax.hold(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.show()