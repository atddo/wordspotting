'''
Created on Jul 13, 2016

@author: da1605
'''

import document_level.main
import numpy as np
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import patch_level.main
import cPickle as pickle
from latent_semantic_indexing.TopicFeatureTransform import TopicFeatureTransform
from latent_semantic_indexing.Weighting import Tf_idf
from scipy.spatial.distance import cdist
from retrieval.main import Retrieval
import os.path
from visualizer import ScoreVisualization
import visualizer

class Word_finder(object):
    '''
    classdocs
    '''


    def __init__(self, sift_step_size, sift_cell_size, sift_n_classes,patch_width,patch_height,patch_hop_size,flatten_dimensions,searchfile,visualize_progress=False, tf_idf = True):
        '''
        Constructor
        '''
        # patch_hop_size = n*sift_step_size
        # patch_width = m*sift_step_size
        # patch_height = i*sift_step_size
        
        self.sift_step_size = sift_step_size
        self.sift_cell_size = sift_cell_size
        self.sift_n_classes = sift_n_classes
        
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_hop_size = patch_hop_size
        self.flatten_dimensions = flatten_dimensions
        self.visualize_progress=visualize_progress
        self.metric = 'cosine'
        self.tf_idf = tf_idf
        
        self.searchfile=searchfile
        filename=searchfile.split('/')[-1].split('.')[0]
        
        image = Image.open(searchfile)
        # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen. 
        im_arr = np.asarray(image, dtype='float32')
        self.im_arr = im_arr
        self.dimensions = im_arr.shape
        dimensions = self.dimensions
        print "Bild-dimensions: "
        print dimensions
            
        pickle_path="pickle/"
        sift_cal_id = self.getIdString("siftcal",[sift_step_size, sift_cell_size, sift_n_classes, filename] )
        labels_id = self.getIdString("labels", [sift_step_size, sift_cell_size, sift_n_classes, filename])
        
        fvd_id = self.getIdString("fvd", [patch_width, patch_height, patch_hop_size, patch_hop_size, sift_n_classes, dimensions[0],dimensions[1]])
        
        tft_id_list = [filename,sift_step_size,sift_cell_size,sift_n_classes,patch_width,patch_height,patch_hop_size,flatten_dimensions]
        transformed_array_id_list = [filename,sift_step_size,sift_cell_size,sift_n_classes,patch_width,patch_height,patch_hop_size,flatten_dimensions]

        
        if tf_idf:
            tft_id_list.append("tf_idf")
            transformed_array_id_list.append("tf_idf")
            
        tft_id = self.getIdString("tft", tft_id_list)
        
        transformed_array_id = self.getIdString("transformed_array", transformed_array_id_list)
        #unique_to_class=[filename,sift_step_size,sift_cell_size,sift_n_classes,patch_width,patch_height,patch_hop_size,flatten_dimensions]
        weighter_id = self.getIdString("weighter", tft_id_list)

        if os.path.isfile(pickle_path+sift_cal_id+".p"):
            print "loading"
            self.siftcalc = pickle.load(open(pickle_path+sift_cal_id +".p","rb"))
            if os.path.isfile(pickle_path+labels_id+".p"):
                print "loading"
                labels = pickle.load(open(pickle_path+labels_id +".p","rb"))
            else:
                _, labels = self.siftcalc.calculate_visual_words_for_document(searchfile, visualize = visualize_progress)
                pickle.dump(labels,open(pickle_path+labels_id +".p","wb"))
                
        else:
            self.siftcalc = document_level.main.SiftCalculator(sift_step_size, sift_cell_size, sift_n_classes)
            _, labels = self.siftcalc.calculate_visual_words_for_document(searchfile, visualize = visualize_progress)
            pickle.dump(self.siftcalc,open(pickle_path+sift_cal_id +".p","wb"))
            pickle.dump(labels,open(pickle_path+labels_id +".p","wb"))
        
        
        
        
        if os.path.isfile(pickle_path+fvd_id+".p"):
            print "loading"
            self.fvd = pickle.load(open(pickle_path+fvd_id +".p","rb"))
        else:
            self.fvd = patch_level.main.feature_vector_descriptor(patch_width, patch_height, patch_hop_size, patch_hop_size, sift_n_classes)

        if (not tf_idf or os.path.isfile(pickle_path+weighter_id+".p")) and os.path.isfile(pickle_path+tft_id+".p") and os.path.isfile(pickle_path+transformed_array_id+".p") and os.path.isfile(pickle_path+fvd_id+".p"):
            print "loading"
            self.tft = pickle.load(open(pickle_path+tft_id +".p","rb"))
            if tf_idf:
                self.weighter = pickle.load(open(pickle_path+weighter_id+".p", "rb"))
            self.transformed_array = pickle.load(open(pickle_path+transformed_array_id +".p","rb"))
        else:
            patch_mat = self.fvd.patch_mat(dimensions[1], dimensions[0], labels, sift_step_size)
            pickle.dump(self.fvd,open(pickle_path+fvd_id +".p","wb"))
            
            pyramid_mat = np.zeros_like(patch_mat)
            for column in range(pyramid_mat.shape[1]):
                for row in range(pyramid_mat.shape[0]):
                    pyramid_mat[row,column] = self.fvd.spatial_pyramid(patch_mat[row,column])
                
            flat_pyramid_mat=np.array([v for i in pyramid_mat for v in i])
            if tf_idf:
                self.weighter = Tf_idf(flat_pyramid_mat)
                flat_pyramid_mat = self.weighter.tf_idf()
                pickle.dump(self.weighter,open(pickle_path+weighter_id +".p","wb"))

            self.tft = TopicFeatureTransform(flatten_dimensions)
            self.tft.estimate(flat_pyramid_mat)
            pickle.dump(self.tft,open(pickle_path+tft_id +".p","wb"))
            self.transformed_array=self.tft.transform(flat_pyramid_mat)
            pickle.dump(self.transformed_array,open(pickle_path+transformed_array_id +".p","wb"))
        print "done"
        
    
    def search(self,groundtrouth):
        #print "SEARCHING STARTS"
        self.visualize_progress = True
        distance_to_end_x = self.dimensions[0]- groundtrouth[2]
        distance_to_end_y = self.dimensions[1]- groundtrouth[3]
        query = (groundtrouth[0] - (self.patch_width - min(distance_to_end_x, self.patch_width)),
                 groundtrouth[1] - (self.patch_height - min(distance_to_end_y, self.patch_height)), 
                 groundtrouth[0] + min(distance_to_end_x, self.patch_width),
                 groundtrouth[1] + min(distance_to_end_y, self.patch_height)
                 )
        query_im = self.im_arr[groundtrouth[1]:groundtrouth[3], groundtrouth[0]:groundtrouth[2]]
        #query_im = self.im_arr[query[1]:query[3], query[0]:query[2]]
        if self.visualize_progress:
            plt.imshow(query_im, cmap=cm.get_cmap('Greys_r'))
            plt.show()
            
        query_sift = self.siftcalc.calculate_visual_words_for_query(query_im, visualize=self.visualize_progress)
        query_pyramid = self.fvd.spatial_pyramid(query_sift)
        if self.tf_idf:
            query_pyramid = self.weighter.tf_idf_query(np.array(query_pyramid))
            query_pyramid = query_pyramid.reshape(query_pyramid.shape[1],)
                
        print query_pyramid.shape
        transformed_query = self.tft.transform(query_pyramid)
        print transformed_query.shape
        print self.transformed_array.shape
        
                
        distances_array = cdist(self.transformed_array, np.array([transformed_query]), metric=self.metric)
        
        distances_mat = distances_array.reshape(self.fvd.getShape())
        vis = visualizer.ScoreVisualization()
        #score_mat_bounds undefined, tuple (x_min,y_min,x_max,y_max) required
        bounds = ((0.5*self.patch_width),(0.5*self.patch_height),(distances_mat.shape[1]*self.patch_hop_size+0.5*self.patch_width),(distances_mat.shape[0]*self.patch_hop_size+0.5*self.patch_height))

        vis.visualize_score_mat(self.searchfile, distances_mat, bounds)
        print (self.patch_width, self.patch_height, self.patch_hop_size, self.searchfile)
        ret = Retrieval(self.patch_width, self.patch_height, self.patch_hop_size, self.searchfile)
        _, non_max_list = ret.non_maximum_suppression(distances_mat)
        coordinates_list = ret.create_list(non_max_list, visualize = self.visualize_progress)
        return coordinates_list

    def getIdString(self,name,unique_to_class):
        ident = ""
        for a in unique_to_class:
            ident= ident + '-' + str(a)
        ident = name + "/" + name + ident
        return ident
        
