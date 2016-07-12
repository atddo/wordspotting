import numpy as np
from scipy.spatial.distance import cdist

class LSIDistance(object):
    """
    Die Klasse fuehrt eine Dimensionsreduzierung mittels LSI Reduzierung durch und berechnet Distanzen von Patches
    zur intern gespeicherten Patchmatrix. Fuer genauere Informationen, siehe Dokumentation der einzelnen Methoden
    """ 
    def __init__(self, n_dims, patchMatrix):
        """Initialisiert die Berechnung der Dimensionsreduzierung.
        
        Params:
            n_dims: Dimension des neuen Raums.
            
            patchMatrix: mit Patches gefuelltes ndarray, zu dem die distances berechnet werden sollen,
                         mit den Dimensionen pN x pM
                         wobei pN die Anzahl der Horizontalen Patches
                         und   pM die Anzahl der Vertikalen Patches angibt
        """
        self.__n_topics = n_dims
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None
        
        self.__patchMatShape = patchMatrix.shape
        oldShape = self.__patchMatShape
        newShape = (oldShape[0]*oldShape[1],oldShape[2])
        patchList = patchMatrix.reshape(newShape)
        
        self.__T, s_arr, self.__D = np.linalg.svd(patchList.T, full_matrices =False)
        #print s_arr

        self.__S_inv =  np.diag(1/s_arr)  
        self.__S = np.diag(s_arr)
        
        d_columns = self.__D.T.shape[0]
        self.__D = self.__D.T[0:d_columns,0:self.__n_topics]
        
        self.__TransformedPatches =self. transform(patchList)

        
    def transform(self, data):
        """Transformiert Daten in den neuen Raum, bestimmt durch die PatchMatrix.
        
        Params:
            data: ndarray, das Patches zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x n_topics).
        """
        transformed = np.dot(data, self.__T)
        transformed = np.dot(transformed, self.__S_inv)
        
        return transformed[:,0:self.__n_topics] 

    def transformedPatchMatrix(self, patchMatrix):
        """ Transformiert die PatchMatrix in den neuen Raum
        
        Params:
            patchMatrix: mit Patches gefuelltes ndarray mit den Dimensionen pN x pM
                         wobei pN die Anzahl der Horizontalen Patches
                         und   pM die Anzahl der Vertikalen Patches angibt
    
        Returns: 
            tfPatchMatrix: ndarray mit den Dimensionen pN x pM, welches transformierte
                           Patches beinhaltet
        
        """
        return self.__TransformedPatches
    
    def getDistanceMatrix(self,  patch, metric='euclidean'):
        """Gibt DistanceMatrix zurueck
        Params:
            patch: Patch dessen Distanz zu den Elementen der Patchmatrix berechnet werden soll
            metric: Distanzmass, welches verwendet werden soll       
        Returns:
            distances: Matrix mit den Distanzen zur PatchMatrix
                         Die Distanz zum Patch an Position x,y in der Patchmatrix
                         befindet sich an Position x,y der Distanzmatrix
       """
        transMat = self.__TransformedPatches
        transPatch = self.transform([patch])[0]
        #print '%s,%s'%(transPatch, transMat[0])
        distances = cdist(transMat, [transPatch], metric=metric)
        distances = distances.reshape(self.__patchMatShape[:2])
        
        return distances
       
def main():
    
    patchMatrix = np.arange(10*6*4).reshape(10,6,4) #PatchMatrix zu welchem die Distanz berechnet werden soll
    
    print 'PatchMatrix:\n%s\n'%patchMatrix 
    
    aPatch = np.arange(4) # irgend ein Patch dessen Distanz zu den Eintraegen in der Patchmatrix berechnet werden soll
    print 'irgend ein Patch:\n%s\n'%aPatch
    
    distCalculator = LSIDistance(3,patchMatrix) # erzeugt einen DistanzCalculator, welcher via LSI eine Dimensionsreduktion(hier auf 3 Dimensionen)
                                                # vornimmt und Distanzen zu den Eintraegen der PatchMatrix berechnen kann
    distanceMatrix = distCalculator.getDistanceMatrix(aPatch) # berechnet Distanzmatrix, welche die Distanzen zwischen den Eintraugen
                                                              # und den Matrixelementen beinhaltet
                                                              # Die Distanz zum Patch an Position x,y in der Patchmatrix
                                                              # befindet sich an Position x,y der Distanzmatrix
                                                        
    print 'Distanzen:\n%s\n'%distanceMatrix
    
    anotherPatch =np.array([ 28,29,30,31])
    
    print 'Distanzen zu %s:\n%s\n'%(anotherPatch,distCalculator.getDistanceMatrix(anotherPatch))

if __name__ == '__main__':
    main()       
    