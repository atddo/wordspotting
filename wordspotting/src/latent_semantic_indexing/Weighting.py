'''
Created on 16.07.2016

@author: Frederik
'''

import numpy as np
class Tf_idf:
    # Zusaetzlich kann man noch die inverse Frequenz von Dokumenten beruecksichtigen
    # in denen ein bestimmter Term vorkommt. Diese Normalisierung wird als  
    # inverse document frequency bezeichnet. Die Idee dahinter ist Woerter die in
    # vielen Dokumenten vorkommen weniger stark im Bag-of-Words Histogramm zu gewichten.
    # Die zugrundeliegende Annahme ist aehnlich wie bei den stopwords (aufgabe1), dass 
    # Woerter, die in vielen Dokumenten vorkommen, weniger Bedeutung fuer die 
    # Unterscheidung von Dokumenten in verschiedene Klassen / Kategorien haben als
    # Woerter, die nur in wenigen Dokumenten vorkommen. 
    # Diese Gewichtung laesst sich statistisch aus den Beispieldaten ermitteln.
    #
    # Zusammen mit der relativen Term Gewichtung ergibt sich die so genannte
    # "term frequency inverse document frequency"
    #
    #                            Anzahl von term in document                       Anzahl Dokumente
    # tfidf( term, document )  = ----------------------------   x   log ( ---------------------------------- ) 
    #                             Anzahl Woerter in document              Anzahl Dokumente die term enthalten
    #
    # http://www.tfidf.com
    #
    # Eklaeren Sie die Formel. Plotten Sie die inverse document frequency fuer jeden 
    # Term ueber dem Brown Corpus.   
    #
    # Implementieren und verwenden Sie die Klasse RelativeInverseDocumentWordFrequecies
    # im features Modul, in der Sie ein tfidf Gewichtungsschema umsetzen.
    # Ermitteln Sie die Gesamt- und klassenspezifischen Fehlerraten mit der Kreuzvalidierung.
    # Vergleichen Sie das Ergebnis mit der absolten und relativen Gewichtung.
    # Erklaeren Sie die Unterschiede in den klassenspezifischen Fehlerraten. Schauen Sie 
    # sich dazu die Verteilungen der Anzahl Woerter und Dokumente je Kategorie aus aufgabe1
    # an. In wie weit ist eine Interpretation moeglich? 
    #                            Anzahl an Index in spat_pyr                       Anzahl spat_pyrs
    # tfidf( index, spat_pyr) = ----------------------------   x   log ( ---------------------------------- ) 
    #                             Gesamtzahl in spat_pyr              Anzahl spat_pyrs die an Index > 0
    def __init__(self, flat_pyramid_mat):
        self.pyramid_mat = flat_pyramid_mat
        self.anzahl_pyrs = float(self.pyramid_mat.shape[0])
        self.spat_pyr_dim = self.pyramid_mat.shape[1]
        
    def tf_idf(self):
        #self.pyramid_mat
        n_in_pyr = np.sum(self.pyramid_mat, axis = 1, dtype = float).reshape(-1,1)
        #print n_in_pyr
        erster_bruch_mat =  np.divide(self.pyramid_mat, n_in_pyr)
        #print erster_bruch_mat
        
        anzahl_pyrs_i_gr_null = np.nansum(np.divide(self.pyramid_mat,self.pyramid_mat), axis = 0)
        #print anzahl_pyrs_i_gr_null
        self.log_mat = np.log(np.divide(self.anzahl_pyrs, anzahl_pyrs_i_gr_null))
        #print log_mat
        self.log_mat[self.log_mat == np.Inf] = 0
        return np.nan_to_num(np.multiply(erster_bruch_mat, self.log_mat))
    
    def tf_idf_query(self, query_mat):
        #print query_mat
        n_in_pyr = np.sum(query_mat, dtype = float).reshape(-1,1)
        #print n_in_pyr
        erster_bruch_mat =  np.divide(query_mat, n_in_pyr)
        #print erster_bruch_mat
        
        
        return np.nan_to_num(np.multiply(erster_bruch_mat, self.log_mat))
        
        
if __name__ == "__main__":
    arr_ = np.array([0,1,0,3,4,0,0,7,0,0,10,0]).reshape(4,3)
    query = np.array([2,1,0])
    print arr_
    weighter = Tf_idf(arr_)
    print weighter.tf_idf()
    print weighter.tf_idf_query(query)
        
        
        