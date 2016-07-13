from word_finder import word_finder
import collections
from visualizer import ScoreVisualization

def roundTo(n,base):
    while n%base != 0:
        n+=1
    return n

# patch_hop_size = n*sift_step_size
# patch_width = m*sift_step_size
# patch_height = i*sift_step_size

sift_step_size = 5
sift_cell_size = 15
sift_n_classes = 100

patch_width = 300
patch_height = 75
patch_hop_size = 25
metric = 'cosine'

flatten_dimensions = 100

visualize_progress=False
searchfile = '../../george_washington_files/2700270.png'


#my_finder = word_finder(sift_step_size, sift_cell_size, sift_n_classes, patch_width, patch_height, patch_hop_size, flatten_dimensions, searchfile, visualize_progress)
print "Initialisierung abgeschlossen"
#my_finder.search((580, 319, 723, 406, "the"))
print "next search"
#my_finder.search((1235, 1518, 1450, 1622, "with"))


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
    positions[word].append((line[0],line[1],line[2],line[3]))



for word in positions.keys():
    print word
    for position in positions[word]:
        query_width = position[2]-position[0]
        width = roundTo(query_width, 50)
        print width
        my_finder = word_finder(sift_step_size, sift_cell_size, sift_n_classes, width, patch_height, patch_hop_size, flatten_dimensions, searchfile, visualize_progress)
        result = my_finder.search(position)
         


    

if __name__ == '__main__':
    pass