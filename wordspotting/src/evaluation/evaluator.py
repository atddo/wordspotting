import numpy as np
from patch_level.main import Overlap_Calculator
import itertools as it

class Evaluator(object):
    
    def __init__(self):
        #do nothin'
        a=0
    
    @staticmethod
    def getHitlist(truth_list, result_list, threshold):
        max_overlap_list = []
        for result in result_list:
            overlap_list = []
            for truth in truth_list:
                overlap = Overlap_Calculator.calculate_overlap(result[0],result[1],result[2],result[3],truth[0],truth[1],truth[2],truth[3])
                overlap_list.append(overlap)
                
            max_o = max(overlap_list)
            
            if max_o < threshold:
                max_overlap_list.append(0)
            else:
                max_overlap_list.append(1)
            

        return max_overlap_list 
    
    @staticmethod
    def calculate_precision(truth_list, result_list, threshold):
        hitlist = Evaluator.getHitlist(truth_list, result_list, threshold)
        
        sum = np.sum(hitlist) # anzahl relevanter Ergebnisse
        
        return sum/float(len(hitlist)) # anz. rel. Erg. in Liste durch anz. Erg. in Liste
    
    @staticmethod
    def calculate_recall(truth_list, result_list, threshold):
        hitlist = Evaluator.getHitlist(truth_list, result_list, threshold)
        
        sum = np.sum(hitlist)
        
        return sum/float(len(truth_list)) # anz. rel. Erg. in Liste durch anzahl rel. Erg. im Datensatz
    
    @staticmethod
    def calculate_avg_precision(truth_list, result_list, threshold):
       
        hitlist = Evaluator.getHitlist(truth_list, result_list, threshold)
        sum = 0
        for a in it.izip(hitlist, np.arange(1,len(hitlist)+1)):
            sub_list = hitlist[:a[1]]
            sum += (a[0]*np.sum(sub_list)) / float(a[1])
            print 'sub_list: %s'%sub_list
            
        return sum / float(len(truth_list))
            
    
    
if __name__ == "__main__":
    print "Test"
    a = [(5,5,10,10),(15,15,14,14),(10,10,14,12)]
    b = [(0,0,6,6), (6,0,7,7), (9,0,11,6), (9,6,11,7), (9,9,11,11), (7,9,8,11), (3,9,6,11), (3,6,6,7),
         (0,0,5,5), (5,0,10,5), (10,10,11,11), (0,10,5,12), (6,0,7,5)]
        
    print 'precision:%s'%Evaluator.calculate_precision(b, a, 0.01)  
    print 'reacall:%s'%Evaluator.calculate_recall(b, a, 0.01)  
    print 'avg precision:%s'%Evaluator.calculate_avg_precision(b, a, 0.01)  
        
        
        