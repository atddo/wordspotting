from word_finder import Word_finder
import collections
from visualizer import ScoreVisualization
from patch_level.main import Overlap_Calculator
from evaluation.evaluator import Evaluator as Eva


def roundTo(n,base):
    while n%base != 0:
        n+=1
    return n
"""
def eval(truth_list, result_list):
    max_overlap_list = []
    for result in result_list:
        overlap_list = []
        for truth in truth_list:
            overlap = Overlap_Calculator.calculate_overlap(result[0],result[1],result[2],result[3],truth[0],truth[1],truth[2],truth[3])
            overlap_list.append(overlap)
        max_overlap_list.append(max(overlap_list))
        
    print "finished!"
    print max_overlap_list
"""
# patch_hop_size = n*sift_step_size
# patch_width = m*sift_step_size
# patch_height = i*sift_step_size

sift_step_size = 5
sift_cell_size = 15
sift_n_classes = 1500

patch_height = 75
patch_hop_size = 20
metric = 'cosine'
threshold = 0.5

flatten_dimensions = 200

visualize_progress=False
tf_idf = False
searchfile = '../../george_washington_files/2700270.png'

page = searchfile.split('/')[-1].split('.')[0]
gt_file = '../../groundtruth/'+page+'.gtp'

with open(gt_file) as f:
    gts = f.readlines()

positions = collections.defaultdict(lambda: [])
for line in gts:
    line = line.split(' ')
    word = line[-1]
    if word.endswith("\n"):
        word = word[:-1]
    positions[word].append((int(line[0]),int(line[1]),int(line[2]),int(line[3])))


eval_list = []
for word in positions.keys():
    if len(positions[word]) > 1:
        for position in positions[word]:
            print "Wort: %s"%word
            query_width = position[2] - position[0]
            query_height = position[3] - position[1]
            if query_height > 3*sift_cell_size:
                width = roundTo(query_width, 20)
                #print width
                #print query_height
                my_finder = Word_finder(sift_step_size, sift_cell_size, sift_n_classes, width, patch_height, patch_hop_size, flatten_dimensions, searchfile, visualize_progress, tf_idf)
                
                result = my_finder.search(position)
                
                recall = Eva.calculate_recall(positions[word], result, threshold)
                precision = Eva.calculate_precision(positions[word], result, threshold)
                avg_precision = Eva.calculate_avg_precision(positions[word], result, threshold)
                eval_list.append((recall, precision, avg_precision))
                print "Wort: %s \nRecall %g \nPrecision %g \navg_precision %g" %(word, recall, precision, avg_precision)
                Eva.calculate_mean(eval_list)
Eva.calculate_mean(eval_list)

    

if __name__ == '__main__':
    pass