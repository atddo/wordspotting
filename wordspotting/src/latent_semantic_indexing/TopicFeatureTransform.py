import numpy as np

class TopicFeatureTransform(object):
    """Realsiert statistische Schaetzung eines Topic Raums und Transformation
    in diesen Topic Raum.
    """ 
    def __init__(self, n_topics):
        """Initialisiert die Berechnung des Topic Raums
        
        Params:
            n_topic: Groesse des Topic Raums, d.h. Anzahl der Dimensionen.
        """
        self.__n_topics = n_topics
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None
        a=1
       
        
    def estimate(self, train_data, train_labels):
        """Statistische Schaetzung des Topic Raums
        
        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
                Hinweis: Fuer den hier zu implementierenden Topic Raum werden die
                Klassenlabels nicht benoetigt. Sind sind Teil der Methodensignatur
                im Sinne einer konsitenten und vollstaendigen Verwaltung der zur
                Verfuegung stehenden Information.
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.__T, s_arr, D = np.linalg.svd(train_data.T, full_matrices =False)
        self.__S_inv =  np.diag(1/s_arr)
        self.__T = np.array([a[:self.__n_topics] for a in self.__T])
        self.__S_inv = [a[:self.__n_topics] for a in self.__S_inv]
        self.__S_inv = np.array(self.__S_inv[:self.__n_topics])
        
      
        
    def transform(self, data):
        """Transformiert Daten in den Topic Raum.
        
        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x n_topics).
        """
        transformed = np.dot(data, self.__T)
        transformed = np.dot(transformed, self.__S_inv)
        return transformed 


    
    
    