import numpy as np
from patch_level.main import Overlap_Calculator
import itertools as it

class Evaluator(object):
    
    @staticmethod
    def calculate_mean(tuple_list):
        recall = 0
        precision = 0
        avg_precision = 0
        n_ = float(len(tuple_list))
        for elem in tuple_list:
            recall += elem[0]
            precision += elem[1]
            avg_precision += elem[2]
        
        print "Mean Recall %g \nMean Precision %g \nMean Average Precision %g" %(recall/n_, precision/n_, avg_precision/n_)
            
     
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
        truth_len = len(truth_list)
        hitlist = Evaluator.getHitlist(truth_list, result_list, threshold)

        sum = 0
        for a in it.izip(hitlist, np.arange(1,len(hitlist)+1)):
            sub_list = hitlist[:a[1]]
            sum += (a[0]*np.sum(sub_list)) / float(a[1])
            #print 'sub_list: %s , %s'%(sub_list,(a[0]*np.sum(sub_list)) / float(a[1]))
            
        return sum / float(truth_len)
    
    
    @staticmethod
    def calculate_mean_avg_precision(truth_lists, result_lists, threshold):
        n=len(result_lists)
        
        sum = 0;
        for t,r in it.izip(truth_lists, result_lists):
            sum += Evaluator.calculate_avg_precision(t,r,threshold)
        
        return sum/n
        
            
    
    
if __name__ == "__main__":
    a = [(1,2,3),(1,4,9)]
    Evaluator.calculate_mean(a)
    print "Test"
    a = [(5,5,10,10),(15,15,14,14),(10,10,14,12)]
    b = [(0,0,6,6), (6,0,7,7), (9,0,11,6), (9,6,11,7), (9,9,11,11), (7,9,8,11), (3,9,6,11), (3,6,6,7),
         (0,0,5,5), (5,0,10,5), (10,10,11,11), (0,10,5,12), (6,0,7,5)]
        
    print 'precision:%s'%Evaluator.calculate_precision(b, a, 0.01)  
    print 'reacall:%s'%Evaluator.calculate_recall(b, a, 0.01)  
    print 'avg precision:%s'%Evaluator.calculate_avg_precision(b, a, 0.01)  
        
        
        