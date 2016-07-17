import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
from scipy.spatial.distance import cdist

class bounding_box_intersection(object):
    
    #
    # Erwartet die Position, H�he und Breite der beiden Patches
    # Returns:  Boolschen Wert ob die beiden Werte kollidieren
    #
  
    def rectangleCollision(self, position1, width1, height1, position2, width2, height2):
        return position1[0] < position2[0] + width2  and 
               position2[0] < position1[0] + width1  and 
               position1[1] < position2[1] + height2 and 
               position2[1] < position1[1] + height1       
    
     