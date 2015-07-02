## RESACL (BPR)

from util import *
from RescalEvaluation import *

train_list = list()
valid_list = list()
test_list = list()
all_triplet_set = set()

exp_tab_dict = dict() # exp table for speed up

param = DD()

def initilizeAll( ):
	'''
	Reset all the global variables
	'''
	global train_list , valid_list, test_list, all_triplet_set, exp_tab_dict, param
	del train_list[:]
	del valid_list[:]
	del test_list[:]
	all_triplet_set.clear()
	exp_tab_dict.clear()
	param = DD()
	

def readTrainTriplet( train_triplet_file, entRel_file_path ):
	'''
	@param train_triplet_file: path of training triplets file
	@param entRel_file_path: entity and relation in training data
	'''
	global train_list, param
	delimiter = param.delimiter
	entity_set = set()
	relation_set = set()
	triplet_list = readTriplet( train_triplet_file )
	for trip in triplet_list:
		curList = trip.split(delimiter)
		assert( len(curList) == 3 )
		h, r, t = curList
		entity_set.add(h)
		entity_set.add(t)
		relation_set.add(r)
		train_list.append( h + delimiter + r + delimiter + t )
	print '#entity:', len(entity_set), '\t#relation:', len(relation_set), '\t#triplets:', len(train_list)
	fout = open(entRel_file_path, 'wb')
	cPickle.dump(entity_set, fout, -1)
	cPickle.dump(relation_set, fout, -1)
	fout.close()
	return train_list, len(entity_set), len(relation_set)


def negativeSampleEntity( h, r, t, filt ):
	'''
	corrupt entity pair
	@param h, r, t: a triplet
	@pram filt: if 1, ensure negative triplet does not exist in train/valid/test sets
	return: negative sample
	'''
	global all_triplet_set, param
	delimiter = param.delimiter
	mxLoop = 1000
	cnt = 0
	en = '-1'
	while True:
		if cnt >= mxLoop:
			return ('-1' + delimiter + r + delimiter + t)
		cnt += 1
		en = np.random.randint(0, param.ent_num)
		if random.random() <= 0.5:
			trip = str(en) + delimiter + r + delimiter + t
			if filt == 1:
				if trip not in all_triplet_set:
					return trip.split(delimiter)
			else:
				return trip.split(delimiter)
		else:
			trip = h + delimiter + r + delimiter + str(en)
			if filt == 1:
				if trip not in all_triplet_set:
					return trip.split(delimiter)
			else:
				return trip.split(delimiter)

def validModel( nv ):
	global emb_ent, emb_rel, valid_list, param
	head_rank_list = list()
	tail_rank_list = list()
	nv = len(valid_list) if nv > len(valid_list) else nv
	for i in xrange(nv):
		h, r, t = valid_list[i].split(param.delimiter)
		ih, ir, it = int(h), int(r), int(t) 
		vh, vr, vt = emb_ent[:,ih], emb_rel[ir,:,:], emb_ent[:,it]
		energy = calRescal( vh, vr, vt)
		
		simiList = list()
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
	# print head_rank_list
	# print tail_rank_list
	hits10 = np.mean( np.asarray(head_rank_list + tail_rank_list) <= 10 )
	meanRank = np.mean(head_rank_list + tail_rank_list)
	return hits10, meanRank
	

def rescalBatchSGD( ):
	global emb_ent, emb_rel, train_list, valid_list, param
	train_size = len( train_list )
	batch_num = train_size/param.batch_size
	best_res = -1e-10
	best_mr = 1e10
	best_epoch = 0
	fw = open(r'RescalBPR-res-lr-'+str(param.lrate) + '-dim-'+str(param.ndim) + '-batch-'+str(param.batch_size) + '-filt-' + str(param.filt)  + '-regul-' + str(param.regul) + '.txt', 'w')
	timeref = time.time()
	for epoch in xrange(1, param.max_epoch+1):
		if epoch%param.validFreq == 0:
			print 'Epoch No.', epoch
		order = np.random.permutation( train_size )
		if param.debug:	print order
		writeEntityEmbedding( emb_ent, 'emb_ent-'+str(epoch)+'.txt' )
		writeRelationEmbedding( emb_rel, 'emb_rel-'+str(epoch)+'.txt' )
		for i in xrange(batch_num+1):
			# timeref = time.time()
			lwbound = i*param.batch_size
			upbound = (i+1)*param.batch_size if train_size > (i+1)*param.batch_size else train_size
			delta_emb_ent = np.zeros( (param.ndim, param.ent_num), dtype=np.float64 )
			delta_emb_rel = np.zeros( (param.rel_num, param.ndim, param.ndim), dtype=np.float64 )
			for j in xrange( lwbound, upbound ):
				if param.debug:	print lwbound, upbound
				cur_list = train_list[ order[j] ].split( param.delimiter )
				if len(cur_list) == 3:
					h, r, t = cur_list
					ih, ir, it = int(h), int(r), int(t) 
					vh, vr, vt = emb_ent[:,ih], emb_rel[ir,:,:], emb_ent[:,it]
					if ( containsNan(vh) or containsNan(vt) ) or containsNan(vr): 
						print 'positive: epoch', epoch, '\th', h, '\tr', r, '\tt', t, '\n'
						return
					if param.debug: print '\n', h, '\t', r, '\t', t, '\n'
					s, delta_h, delta_r, delta_t = calculateGradient( vh, vr, vt, 1)
					if ( np.isnan(s) or ( containsNan(delta_h) or containsNan(delta_t) ) ) or containsNan(delta_r):
						print 'positive gradient: epoch', epoch, '\th', h, '\t', r, '\t', t, '\n'
						print '\tscore:', s, '\tdelta_h:\t', containsNan(delta_h), '\tdelta_r:', containsNan(delta_r), '\tdelta_t:', containsNan(delta_t)
						return
					if param.debug:
						print h, '\n', vh, '\n', r, vr, '\n', t, vt
						print '\tscore:', s, '\tdelta_h:\t', delta_h, '\n\tdelta_r:', delta_r, '\n\tdelta_t:', delta_t
					if param.checkGradient:
						maxDiff = checkNumericalGradient(vh, vr, vt, 1)
						if param.debug: print 'maxDiff', maxDiff
						if (maxDiff > 1e-5):
							print 'Failed passing gradient checking....', maxDiff
					delta_emb_ent[:,ih] += delta_h
					delta_emb_rel[ir,:,:] += delta_r
					delta_emb_ent[:,it] += delta_t
					
					hn, rn, tn = negativeSampleEntity( h, r, t, param.filt )
					if hn != '-1':
						ihn, irn, itn = int(hn), int(rn), int(tn)
						vhn, vrn, vtn = emb_ent[:,ihn], emb_rel[irn,:,:], emb_ent[:,itn]
						if ( containsNan(vhn) or containsNan(vtn) ) or containsNan(vrn): 
							print 'negative: epoch', epoch, '\thn', hn, '\trn', rn, '\ttn', tn, '\n'
							return
						sn, delta_hn, delta_rn, delta_tn = calculateGradient(vhn, vrn, vtn, 0)
						if ( np.isnan(sn) or ( containsNan(delta_hn) or containsNan(delta_tn) ) ) or containsNan(delta_rn): 
							print 'negative gradient: epoch', epoch,'\tscore:', sn, '\tdelta_hn:\t', containsNan(delta_hn), '\tdelta_rn:', containsNan(delta_rn), '\tdelta_tn:', containsNan(delta_tn)
							return
						if param.debug:
							print 'corrupt entity: hn:', hn, '\t r:', r, '\t tn:', tn
							print '\tscore:', sn, '\tdelta_hn:\t', delta_hn, type(delta_hn), '\n\tdelta_r:', delta_rn, '\n\tdelta_tn:', delta_tn
						if param.checkGradient:
							maxDiff = checkNumericalGradient(vhn, vrn, vtn, 0)
							if param.debug: print 'maxDiff', maxDiff							
							if (maxDiff > 1e-3):
								print 'Failed passing gradient checking....', maxDiff
						delta_emb_ent[:,ihn] += delta_hn
						delta_emb_rel[irn,:,:] += delta_rn
						delta_emb_ent[:,itn] += delta_tn
			# print '\t time per-batch:', round( time.time() - timeref, 3 )
			# timeref = time.time()
			# update embedding
			delta_emb_ent =  delta_emb_ent + param.regul * emb_ent
			delta_emb_rel =  delta_emb_rel + param.regul * emb_rel
			if param.debug:
				print '\n=============================\nentity gradient embedding:\n'
				for i in xrange(param.ent_num):
					print i, '\n', delta_emb_ent[:,i]
				print '\nrelation gradient embedding:\n'
				for i in xrange(param.rel_num):
					print i, '\n', delta_emb_rel[i,:,:]
			
			emb_ent -= param.lrate * delta_emb_ent# minimize
			emb_rel -= param.lrate * delta_emb_rel# minimize
			if param.normFlag:
				emb_ent = emb_ent / np.sqrt( np.sum( emb_ent ** 2, axis=0 ) )
				for i in xrange(param.rel_num):
					emb_rel[i,:,:] = normilizeRelation(emb_rel[i,:,:])
			if param.debug:
				print 'embedding after update:\nentity:'
				for i in xrange(param.ent_num):
					print i, '\n', emb_ent[:,i]
				print '\n\nrelation embedding:\n'
				for i in xrange(param.rel_num):
					print i, '\n', emb_rel[i,:,:]
				print '\n==========================================\n\n'
		# print 'Epoch No.', epoch, '\t time per-epoch:', round( time.time() - timeref, 3 )
		# timeref = time.time()
		# save current model
		if epoch % param.validFreq == 0:
			time_per_epoch = round(time.time() - timeref, 3) / param.validFreq
			print '\ttime_per_epoch', time_per_epoch
			model_file_path = 'model' + '/' + str(epoch) + '-model.pkl'
			fout = open(model_file_path, 'wb')
			cPickle.dump(emb_ent, fout, -1)
			cPickle.dump(emb_rel, fout, -1)
			fout.close()
			timeref = time.time()
			hits10, meanRank = validModel( param.numValid )
			valid_time = round(time.time() - timeref, 3)
			if (hits10 > best_res) or (hits10 == best_res and best_mr > meanRank):
				best_res = hits10
				best_mr = meanRank
				best_epoch = epoch
			print '\tvalid-time:%.1f \t hits10:%.4f \t meanRank: %.2f \t Best-Hits@10:%.4f \t Best-Epoch:%d' %(valid_time, hits10, meanRank, best_res, best_epoch)
			fw.write( 'Epoch:%d \t valid-time:%.1f \t hits10:%.4f \t meanRank: %.2f \t Best-Hits@10:%.4f \t Best-Epoch:%d\n' %( epoch, valid_time, hits10, meanRank, best_res, best_epoch) )
			fw.flush()
			timeref = time.time()
	fw.flush()
	fw.close()
	return best_epoch

def solve( conf_file_path ):
	global param, emb_ent, emb_rel, exp_tab_dict, train_list, valid_list, test_list, all_triplet_set
	
	initilizeAll()
	param = readConf( conf_file_path ) # load in configuration file
	print param
	exp_tab_dict = calExpTable(param.EXP_TABLE_SIZE, param.MAX_EXP) # approximate exp table
	
	train_list, param.ent_num, param.rel_num = readTrainTriplet( param.train_file_path, param.entRel_file_path )
	valid_list = readTriplet( param.valid_file_path )
	test_list = readTriplet( param.test_file_path )
	all_triplet_set = set( train_list + valid_list + test_list )
	
	np.random.seed( param.randomSeed )
	emb_ent, emb_rel = initilizeEmbedding( np.random, param.ndim, param.ent_num, param.rel_num ) # !! After Read In Traning Data
	best_epoch =  rescalBatchSGD( )
	return best_epoch

	
if __name__ == '__main__':
	conf_file_path = 'rescalBPR.conf'
	epoch = solve(conf_file_path)
	print 'Best Epoch:', epoch
	emb_file_path = 'model/' + str(epoch) + '-model.pkl'
	print evaluationModel(conf_file_path, emb_file_path)
	



