## RESACL (BPR)

import os
import sys
import time
import copy
import cPickle
import warnings

import ConfigParser
import random
import numpy as np


class DD(dict):
	"""This class is only used to replace a state variable of Jobman"""

	def __getattr__(self, attr):
		if attr == '__getstate__':
			return super(DD, self).__getstate__
		elif attr == '__setstate__':
			return super(DD, self).__setstate__
		elif attr == '__slots__':
			return super(DD, self).__slots__
		return self[attr]

	def __setattr__(self, attr, value):
		assert attr not in ('__getstate__', '__setstate__', '__slots__')
		self[attr] = value

	def __str__(self):
		return 'DD%s' % dict(self)

	def __repr__(self):
		return str(self)

	def __deepcopy__(self, memo):
		z = DD()
		for k, kv in self.iteritems():
			z[k] = copy.deepcopy(kv, memo)
		return z
	
	
def containsNan( ay ):
	if np.isnan( np.min(ay) ):
		return 1
	return 0


def readConf( conf_file_path ):
	cf = ConfigParser.ConfigParser()
	cf.read(conf_file_path)
	param = DD()
	param.train_file_path = cf.get('dataset','train_file_path')
	param.valid_file_path = cf.get('dataset','valid_file_path')
	param.test_file_path = cf.get('dataset','test_file_path')
	param.entRel_file_path = cf.get('dataset','entRel_file_path')
	param.ent_num = int( cf.get('dataset','ent_num') )
	param.rel_num = int( cf.get('dataset','rel_num') )
	param.delimiter = '\t' # cf.get('dataset','delimiter')
	
	param.MAX_EXP =  int( cf.get('expTable','MAX_EXP') )
	param.EXP_TABLE_SIZE =  int( cf.get('expTable','EXP_TABLE_SIZE') )
	
	param.debug = int( cf.get('debug','debug') )
	param.checkGradient = int( cf.get('debug','checkGradient') )
	
	param.numValid = int( cf.get('valid','numValid') )
	param.validFreq = int( cf.get('valid','validFreq') )
	
	param.ndim = int( cf.get('model','ndim') )
	param.max_epoch = int( cf.get('model','max_epoch') )
	param.batch_size = int( cf.get('model','batch_size') )
	param.filt = int( cf.get('model','filt') )
	param.randomSeed = int( cf.get('model','randomSeed') )
	param.normFlag = int( cf.get('model','normFlag') )
	param.lrate = float( cf.get('model','lrate') )
	param.regul = float( cf.get('model','regul') )
	param.a = float( cf.get('model','a') )
	param.b = float( cf.get('model','b') )
	return param
	
	
def calExpTable(EXP_TABLE_SIZE, MAX_EXP):
	exp_tab_dict = dict()
	for i in xrange(EXP_TABLE_SIZE+1):
		exp_tab_dict[i] = np.exp(  ( 2.*i/EXP_TABLE_SIZE - 1 ) * MAX_EXP )
		exp_tab_dict[i] = exp_tab_dict[i] / ( 1 + exp_tab_dict[i] ) # 1/( 1+e(-x) ) == e(x)/( 1+e(x) )
	return exp_tab_dict

		
def readTriplet( triplet_file ):
	'''
	@param triplet_file: path of triplets file
	return: triplet list
	'''
	triplet_list = list()
	with open( triplet_file, 'r' ) as fr:
		for line in fr:
			line = line.strip('\r').strip('\n')
			triplet_list.append( line )
	return triplet_list

def normilizeEntity( mat ):
	return mat / np.sqrt( np.sum( mat ** 2, axis=0 ) )
	
def normilizeRelation( mat ):
	r,c = mat.shape
	cnt = np.sqrt( np.sum( mat ** 2 ) )
	mat /= cnt
	return mat
	
def initilizeEmbedding( rng, d, ent_num, rel_num ):
	'''
	initialize embedding of entity and relation, all are normlized to 1
	@param rng: numpy.random module for number generation.
	@param d: dimension of embedding
	'''
	
	# wbound = np.sqrt( 6./d )
	wbound = 1
	
	emb_ent = np.array( rng.uniform( low = -wbound, high = wbound, size = (d,ent_num) ), dtype=np.float64 ) # entity as vector of R^{d}
	emb_ent = normilizeEntity( emb_ent )
	# print id, emb_dict[id]
	
	emb_rel = np.array( rng.uniform( low = -wbound, high = wbound, size = (rel_num,d,d) ), dtype=np.float64 )
	for i in xrange(rel_num):
		emb_rel[i,:,:] = normilizeRelation( emb_rel[i,:,:] )
		# print id, emb_dict[id]
	return emb_ent, emb_rel

def writeEntityEmbedding(emb_ent, emb_file):
	r, c = emb_ent.shape
	with open(emb_file, 'w') as fw:
		for i in xrange(c):
			fw.write( str(i) + ':' )
			for j in xrange(r):
				fw.write( ' ' + str(emb_ent[j,i]) )
			fw.write('\n')

def writeRelationEmbedding(emb_rel, emb_file):
	r, d, d = emb_rel.shape
	with open(emb_file, 'w') as fw:
		for i in xrange(r):
			fw.write( str(i) + ':\n' )
			fw.write( str(emb_rel[i,:,:]) + '\n\n' )
	
def calRescal( vh, vr, vt):
	hTr = np.dot(vh, vr) # vh.transpose() * vr  h^T r
	hTrt = np.dot(hTr, vt) # ( hTr * vt ).flat[0]
	return hTrt

def calSigmaRescal( vh, vr, vt):
	hTr = np.dot(vh, vr) # vh.transpose() * vr  h^T r
	hTrt = np.dot(hTr, vt) # ( hTr * vt ).flat[0]
	x = hTrt
	# print '\thTr:', hTr, '\nhTrt:', hTrt
	# print vh.shape, vr.shape, vt.shape, hTr.shape, hTrt.shape, x.shape, type(x)
	# return 1./( 1 + np.exp(-x) )
	if x > 20: # param.MAX_EXP:
		return 1.
	elif x < -20:
		return 0.
	else:
		return 1./( 1 + np.exp(-x) )# exp_tab_dict[ int( (x + param.MAX_EXP) * (param.EXP_TABLE_SIZE / param.MAX_EXP / 2) ) ]

			
def calculateGradient(vh, vr, vt, y):
	try:
		s = calSigmaRescal(vh, vr, vt)
		delta_h = ( s - y ) * np.dot(vr, vt) # Rt
		delta_r = ( s - y ) * np.outer(vh, vt) # h^{T}t
		delta_t = ( s - y ) * np.dot(vh, vr) # R^{T}h
	except:
		print 'calculateGradient\tscore:', s, vh.shape, vr.shape, vt.shape
		# print 'vh:\t', vh, '\nvt:', vt, '\nvr:', vr
		# print '\tscore:', s, '\ty:', y,'\tdelta_h:\t', delta_h, '\tdelta_t:\t', delta_t,'\tdelta_r:\t', delta_r
	
	return s, delta_h, delta_r, delta_t


def negativeLogLikehood(vh, vr, vt, y):
	s = calSigmaRescal(vh, vr, vt)
	return -y*np.log(s) - (1-y)*np.log(1-s)
	
def checkNumericalGradient(vh, vr, vt, y):
	s, delta_h, delta_r, delta_t = calculateGradient(vh, vr, vt, y)
	cost = negativeLogLikehood(vh, vr, vt, y)
	mu = 1e-5
	maxDiff = 0
	vh_v = vh
	tmpdelta = 0
	# print 'cost:', cost
	for k in xrange(0, 20):
		i = random.randint(0, len(vh_v)-1)
		vh_v[i] = vh_v[i] + mu
		val = negativeLogLikehood(vh_v, vr, vt, y)
		# print 'vh_v:', vh_v, '\tval:', val
		tmpdelta = (val - cost) / mu
		vh_v[i] = vh_v[i] - mu
		# print 'tmpdelta:', tmpdelta, '\tdelta_h[i]:', delta_h[i],'\tdiff:',np.abs(delta_h[i] - tmpdelta)
		maxDiff = max( maxDiff, np.abs(delta_h[i] - tmpdelta) )
	# print 'h', maxDiff
	vt_v = vt
	for k in xrange(1, 20):
		i = random.randint(0, len(vt_v)-1)
		vt_v[i] = vt_v[i] + mu
		val = negativeLogLikehood(vh, vr, vt_v, y)
		tmpdelta = (val - cost) / mu
		vt_v[i] = vt_v[i] - mu
		# print np.abs(delta_t[i] - tmpdelta)
		maxDiff = max( maxDiff, np.abs(delta_t[i] - tmpdelta) )
	return maxDiff

def getRank( val, simiList, asend = True ):
	"""
	"""
	cnt = 0
	cntSame = 0
	for i in xrange( len( simiList ) ):
		if asend == True and simiList[i] > val:
			cnt += 1
		if asend == False and simiList[i] < val:
			cnt += 1
		elif simiList[i] == val:
			cntSame += 1
	# print cnt, cntSame
	return cnt + ( cntSame + 1)*1.0/2
