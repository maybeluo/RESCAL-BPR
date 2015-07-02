## Mini-batch SGD
## Test: triplet classification

from util import *


exp_tab_dict = dict() # exp table for speed up
param = DD()

def loadEntityRelation( entRel_file_path ):
    entity_set = set()
    relation_set = set()
    fin = open(entRel_file_path, 'rb')
    entity_set = cPickle.load( fin )
    relation_set = cPickle.load( fin )
    fin.close()
    return entity_set, relation_set
    
def loadEmbedding( pkl_file_path ):
    fin = open(pkl_file_path, 'rb')
    emb_ent = cPickle.load( fin )
    emb_rel = cPickle.load( fin )
    fin.close()
            
    return emb_ent, emb_rel

def evaluationModel( conf_file_path, emb_file_path ):
    param = readConf( conf_file_path ) # load in configuration file
    emb_ent, emb_rel = loadEmbedding(emb_file_path)
    entity_set, relation_set = loadEntityRelation( param.entRel_file_path )
    test_list = readTriplet( param.test_file_path )
    head_rank_list = list()
    tail_rank_list = list()
    nv = len(test_list)
    for i in xrange(nv):
        h, r, t = test_list[i].split(param.delimiter)
        ih, ir, it = int(h), int(r), int(t) 
        vh, vr, vt = emb_ent[:,ih], emb_rel[ir,:,:], emb_ent[:,it]
        energy = calRescal( vh, vr, vt)
        
        simiList = list()
        t_mluti_r = np.dot( vt, vr.T )
        simiList = np.dot(t_mluti_r, emb_ent).tolist()
        head_rank = getRank(energy, simiList)
        
        simiList = list()
        h_mluti_r = np.dot( vh, vr )
        simiList = np.dot(h_mluti_r, emb_ent).tolist()
        tail_rank = getRank(energy, simiList)
        
        head_rank_list.append( head_rank )
        tail_rank_list.append( tail_rank )
    print head_rank_list
    print tail_rank_list
    hits10 = np.mean( np.asarray(head_rank_list + tail_rank_list) <= 10 )
    meanRank = np.mean(head_rank_list + tail_rank_list)
    return hits10, meanRank
    
if __name__ == '__main__':
    epoch = 10
    conf_file_path = 'rescalBPR.conf'
    emb_file_path = 'model/' + str(epoch) + '-model.pkl'
    print evaluationModel(conf_file_path, emb_file_path)
    
    
