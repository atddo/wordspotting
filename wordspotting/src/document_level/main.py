
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
import matplotlib
# ACHTUNG: Die vlfeat Python Bindungs werden nur fuer Linux unterstuetzt und 
# muessen nicht unbedingt eingebunden werden --> siehe unten
import vlfeat
import cPickle as pickle
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from scipy import signal
from scipy import misc
#from visualization import hbar_plot, bar_plot
from collections import defaultdict


def visual_word_description(step_size = 150, cell_size = 3, visualize = True):

   
    #
    document_image_filename = '../../george_washington_files/2710271.png'
    image = Image.open(document_image_filename)
    # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen. 
    im_arr = np.asarray(image, dtype='float32')
    # Die colormap legt fest wie die Intensitaetswerte interpretiert werden.
    if visualize:
        plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
        plt.show()
    
    
    
    # SIFT Deskriptoren berechnen
    frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
#     pickle_densesift_fn = '2700270-small_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
#     frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    frames = frames.T
    desc = desc.T

    # 
    # Um eine Bag-of-Features Repraesentation des Bilds zu erstellen, wird ein
    # Visual Vocabulary benoetigt. Im Folgenden wird es in einer Clusteranalyse
    # berechnet. Fuer die Clusteranalyse wird Lloyd's Version des k-means Algorithmus
    # verwendet. Parameter sind
    # - die Anzahl der Centroiden in der Clusteranalyse (n_centroids). Das entspricht 
    # der Groesse des Visual Vocabulary bzw. der Anzahl von Visual Words. 
    # - Der Anzahl von Durchlaeufen des Algorithmus (iter)
    # - Der Initialisierung (minit). Der Wert 'points' fuehrt zu einer zufaelligen
    #   Auswahl von gegebenen Datenpunkten, die als initiale Centroiden verwendet
    #   werden.
    # Die Methode gibt zwei NumPy Arrays zurueck: 
    #  - Das sogenannte Codebuch. Eine zeilenweise organisierte Matrix mit Centroiden (hier nicht verwendet).
    #  - Einen Vektor mit einem Index fuer jeden Deskriptor in desc. Der Index bezieht
    #    sich auf den aehnlichsten Centroiden aus dem Codebuch (labels).
    #
    # Die Abbildung von Deskriptoren auf Centroiden (Visual Words) bezeichnet man als Quantisierung.
    n_centroids = 40 
    _, labels = kmeans2(desc, n_centroids, iter=20, minit='points')

    #
    # Die Deskriptoren und deren Quantisierung werden nun visualisiert. Zu jedem 
    # Deskriptor werden dazu die Mittelpunkte und die 4x4 Zellen eingezeichnet.
    # Die Farbe des Mittelpunkts codiert den Index des Visual Words im Visual Vocabulary
    # (Codebuch). Beachten Sie, dass durch diese Kodierung einige Farben sehr 
    # aehnlich sein koennen. 
    # Da das Zeichnen der 4x4 Zellen fuer jeden Deskriptor viel Performance kosten
    # kann, ist es moeglich es ueber das Flag draw_descriptor_cells abzuschalten.
    #
    if visualize:
        draw_descriptor_cells = True
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
        ax.hold(True)
        ax.autoscale(enable=False)
        colormap = cm.get_cmap('jet')
        desc_len = cell_size * 4
        for (x, y), label in zip(frames, labels):
            color = colormap(label / float(n_centroids))
            circle = Circle((x, y), radius=1, fc=color, ec=color, alpha=1)
            rect = Rectangle((x - desc_len / 2, y - desc_len / 2), desc_len, desc_len, alpha=0.08, lw=1)
            ax.add_patch(circle)
            if draw_descriptor_cells:
                for p_factor in [0.25, 0.5, 0.75]:
                    offset_dyn = desc_len * (0.5 - p_factor)
                    offset_stat = desc_len * 0.5
                    line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08, lw=1)
                    line_v = Line2D((x - offset_dyn , x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08, lw=1)
                    ax.add_line(line_h)
                    ax.add_line(line_v)
            ax.add_patch(rect)
        
        plt.show()
    
    
    
    
    



if __name__ == '__main__':
    visual_word_description();
    
    