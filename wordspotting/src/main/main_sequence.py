from word_finder import Word_finder
import collections
from visualizer import ScoreVisualization
from patch_level.main import Overlap_Calculator

def roundTo(n,base):
    while n%base != 0:
        n+=1
    return n

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
# patch_hop_size = n*sift_step_size
# patch_width = m*sift_step_size
# patch_height = i*sift_step_size

sift_step_size = 5
sift_cell_size = 15
sift_n_classes = 100

patch_height = 75
patch_hop_size = 10
metric = 'cosine'

flatten_dimensions = 100

visualize_progress=False
tf_idf = True
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
for word in ["they"]:
    print word
    if len( positions[word])>1:
        for position in positions[word]:
            query_width = position[2] - position[0]
            query_height = position[3] - position[1]
            width = query_width #roundTo(query_width, 50)
            #print width
            #print query_height
            my_finder = Word_finder(sift_step_size, sift_cell_size, sift_n_classes, width, query_height, patch_hop_size, flatten_dimensions, searchfile, visualize_progress, tf_idf)
            
            result = my_finder.search(position)
            eval_list.append(eval(positions[word],result))
             


    

if __name__ == '__main__':
    pass