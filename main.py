import threading
import tensorflow as tf
import numpy as np
import random, sys
from PIL import Image, ImageEnhance
from scipy.misc import imread, imresize
import glob, pickle
from random import shuffle
from model_vgg import model_vgg
import time
from tqdm import tqdm
from utils import *
import os
import argparse
from q_network import q_network, q_network_shallow, q_network_large
from get_hist import *
from get_global_feature import get_global_feature
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, linewidth=200, threshold=np.nan)
from action import take_action, action_size

class Agent:
	def __init__(self, prefix):
		""" init the class """
		self.save_raw = True
		self.finetune = False
		self.logging = False
		self.use_history = True
		self.use_batch_norm = False
		self.deep_feature_model_path = "./vgg16_pretrain.npy"	

		self.enqueue_repeat = 3
		self.prep	= None
		self.double_q = True
		self.duel = False
		self.use_deep_feature = True
		self.use_color_feature = True
		self.deep_feature_len = 0
		if self.use_deep_feature:
			self.deep_feature_len = 4096
		if not self.use_color_feature:
			self.color_type = 'None'
			self.color_feature_len = 0
		else:
			self.color_type = 'lab_8k'
			self.color_feature_len = 8000

		self.w = {}
		self.batch_size = 4
		self.max_step = 3000000
		self.seq_len = 50
		self.feature_length = self.deep_feature_len + self.color_feature_len + self.use_history*self.seq_len*action_size
		
		self.q = None
		self.q_w = None
		self.target_q = None
		self.target_q_w = None
		self.target_q_w_input = None # input tensor for copying current network to target network
		self.target_q_w_assign_op = None # operation for copying current network to target network
		self.delta = None
		self.min_delta = -1.0
		self.max_delta = 1.0
		self.action_size = action_size
		with tf.variable_scope('step'):
			self.step_op = tf.Variable(0, trainable=False, name='step')
			self.step_input = tf.placeholder('int32', None, name='step_input')
			self.step_assign_op = self.step_op.assign(self.step_input)  # is it necessary?
		self.max_reward = 0.5
		self.min_reward = -0.5
		self.memory_size = 5000
		self.target_q_update_step = 1000
		self.test_step = 160000
		self.learning_rate_minimum = 0.00000001
		self.learning_rate = 0.00001
		self.learning_rate_decay_step = 5000
		self.learning_rate_decay = 0.96
		self.learn_start = 100
		self.train_frequency = 1
		self.discount = 0.95
		self.save_interval = self.test_step
		self.prefix = prefix

		### MIT5K C
		self.data_dir	= '/hdd2/DATA/MIT5K/C/'
		self.test_count = 63

		self.train_dir	= os.path.join(self.data_dir, 'train')
		self.test_dir	= os.path.join(self.data_dir, 'test')

		self.num_thread = 5
		self.config = {
					"learning_rate"				: self.learning_rate, 
					"learning_rate_decay_rate"	: self.learning_rate_decay, 
					"learning_rate_minimum"		: self.learning_rate_minimum,
					"target_q_update_step"		: self.target_q_update_step,
					"discount"					: self.discount,
					"min_reward"				: self.min_reward,
					"max_reward"				: self.max_reward,
					"min_delta"					: self.min_delta,
					"max_delta"					: self.max_delta,
					"feature_type"				: self.color_type,
					"seq_len"					: self.seq_len,
					"batch_size"				: self.batch_size,
					"prefix"					: self.prefix,
					"memory_size"				: self.memory_size,
					"action_size"				: action_size,
				}

	def init_img_list(self, img_list_path=None, test=False):
		if img_list_path:
			with open(img_list_path, 'rb') as f:
				self.img_list = pickle.load(f)
			self.train_img_list = self.img_list['train_img_list']
			self.test_img_list 	= self.img_list['test_img_list']
		else:
			self.train_img_list	= glob.glob(os.path.join(self.train_dir, "raw/*.jpg"))
			self.test_img_list	= glob.glob(os.path.join(self.test_dir, "raw/*.jpg"))
			self.img_list = {'train_img_list': self.train_img_list, 'test_img_list': self.test_img_list }

			with open('./test/'+self.prefix+'/img_list.pkl', 'wb') as f:
				pickle.dump(self.img_list, f)
		with open('./test/'+self.prefix+'/config', 'wb') as f:
			pickle.dump(self.config, f)
		self.train_img_count = len(self.train_img_list)
		self.test_img_count = len(self.test_img_list)
			
	def save_model(self, step=None):
		print (" [*] Saving checkpoints...")
		model_name = type(self).__name__
		self.saver.save(sess, "./checkpoints/"+self.prefix+".ckpt", global_step=step)
	def load_model(self, model_path):
		self.saver.restore(sess, model_path)
		print (" [*] Load success: %s"%model_path)
		return True

	
	def train_with_queue(self):
		self.step=0
		# QUEUE loading setup
		self.coord = tf.train.Coordinator()
		for i in range(self.num_thread):
			t = threading.Thread(target=enqueue_samples, args=(self, self.coord))
			t.start()

		# RUN queue update
		for self.step in tqdm(range(0, self.max_step), ncols=70, initial=0):
			if self.step==0:
				# dry test
				print "do dry test"
				self.test(idx=0)

			self.q_learning_minibatch()
			if self.step % self.target_q_update_step == self.target_q_update_step-1:
				print "update"
				self.update_target_q_network()

			if self.step%self.save_interval == self.save_interval-1:
				self.save_model(self.step+1)
			if self.step%self.test_step == self.test_step-1:
				init_scores		= []
				final_scores	= []
				for i in range(self.test_count):
					#init_score, final_score = self.test()
					init_score, final_score = self.test(in_order=True, idx=i)
					init_scores.append(init_score)
					final_scores.append(final_score)
				init_scores = [str(v) for v in init_scores]
				final_scores = [str(v) for v in final_scores]
				with open('./test/'+self.prefix+'/test.txt', 'a') as f:
					f.write("step %d\n"%self.step)
					f.write(" ".join(init_scores)+"\n")
					f.write(" ".join(final_scores)+"\n")


	def q_learning_minibatch(self, sample_terminal=False):
		state, reward, action, next_state, terminal = sess.run([self.s_t_many, self.reward_many, self.action_many, self.s_t_plus_1_many, self.terminal_many])

		if self.double_q:
			if self.use_batch_norm:
				q, q_actions = sess.run([self.q, self.q_action], feed_dict={self.s_t: state, self.phase_train: True})
			else:
				q, q_actions = sess.run([self.q, self.q_action], feed_dict={self.s_t: state})
			q_t_plus_1 		= self.target_q.eval({self.target_s_t: next_state}, session=sess)
			pred_action = q_actions
			max_q_t_plus_1 = []
			for i in range(self.batch_size):
				max_q_t_plus_1.append(q_t_plus_1[i][pred_action[i]])
			max_q_t_plus_1 = np.reshape(np.array(max_q_t_plus_1), [self.batch_size,1])
            
			terminal = np.array(terminal) + 0.
			target_q_t = (1-terminal) * self.discount * max_q_t_plus_1 + reward
			action = np.reshape(action, target_q_t.shape)
		else:
			q_t_plus_1 		= self.target_q.eval({self.target_s_t: next_state}, session=sess)
			terminal_np 	= np.array(terminal) + 0.
			max_q_t_plus_1	= np.reshape(np.max(q_t_plus_1, axis=1), terminal_np.shape) # next state's maximum q value
			target_q_t		= (1-terminal_np)* self.discount * max_q_t_plus_1 + reward
			action = np.reshape(action, target_q_t.shape)
		action = np.reshape(action, (self.batch_size))
		target_q_t = np.reshape(target_q_t, (self.batch_size))
		if self.use_batch_norm:
			_, q_t, loss, delta, one_hot = sess.run([self.optim, self.q, self.loss, self.delta, self.action_one_hot], 
				feed_dict={	self.target_q_t	: target_q_t, 
							self.action		: action, 
							self.s_t		: state,
							self.phase_train: True})
		else:
			_, q_t, loss, delta, one_hot = sess.run([self.optim, self.q, self.loss, self.delta, self.action_one_hot], 
				feed_dict={	self.target_q_t	: target_q_t, 
							self.action		: action, 
							self.s_t		: state })
		if self.logging:
			print "q_t"
			print q_t
			print "delta"
			print delta
			print "terminal"
			print terminal



	def init_model(self,model_path=None):
		""" load the Policy Network """
		##################
		# with RandomShuffleQueue as the replay memory
		##################
		self.s_t_single			= tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_single")
		self.s_t_plus_1_single	= tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_plus_1_single")
		self.action_single		= tf.placeholder('int64', 	[self.batch_size, 1], name='action_single')
		self.terminal_single	= tf.placeholder('int64', 	[self.batch_size, 1], name='terminal_single')
		self.reward_single		= tf.placeholder('float32', [self.batch_size, 1], name='reward_single')


		self.s_t		= tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t")
		self.target_q_t	= tf.placeholder('float32', [self.batch_size], name='target_q_t')
		self.action		= tf.placeholder('int64', [self.batch_size], name='action')


		self.queue = tf.RandomShuffleQueue(self.memory_size, 1000, [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64], [[self.feature_length], [self.feature_length], 1, 1, 1])
		self.enqueue_op = self.queue.enqueue_many([self.s_t_single, self.s_t_plus_1_single, self.reward_single, self.action_single, self.terminal_single])

		self.s_t_many, self.s_t_plus_1_many, self.reward_many, self.action_many, self.terminal_many = self.queue.dequeue_many(self.batch_size)

		self.target_s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="target_s_t")

		if self.use_batch_norm:
			self.phase_train = tf.placeholder(tf.boolean,name='phase_train')
			self.q, self.q_w = q_network(self.s_t, 'pred', input_length=self.feature_length, num_action=self.action_size, duel=self.duel, batch_norm=self.use_batch_norm, phase_train=self.phase_train)#resulting q values...
			self.target_q, self.target_q_w = q_network(self.target_s_t, 'target', input_length=self.feature_length, num_action=self.action_size, duel=self.duel, batch_norm=self.use_batch_norm, phase_train=self.phase_train)#resulting q values....
		else:
			self.q, self.q_w = q_network(self.s_t, 'pred', input_length=self.feature_length, num_action=self.action_size, duel=self.duel)#resulting q values...
            
			# build network.......
			self.target_q, self.target_q_w = q_network(self.target_s_t, 'target', input_length=self.feature_length, num_action=self.action_size, duel=self.duel)#resulting q values....
		self.q_action = tf.argmax(self.q, dimension=1) # is dimension really 1??
		self.target_q_action = tf.argmax(self.target_q, dimension=1)


		# optimizer
		

		with tf.variable_scope('optimizer'):
			self.action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
			print "self.q shape"
			print self.q.get_shape().as_list()
			print "self.action_one_hot  shape"
			print self.action_one_hot.get_shape().as_list()
			q_acted				= tf.reduce_sum(self.q*self.action_one_hot, reduction_indices=1, name='q_acted')
			print "q_acted shape"
			print q_acted.get_shape().as_list()
			print "target_q_t shape"
			print self.target_q_t.get_shape().as_list()
			self.delta			= self.target_q_t - q_acted
			print "delta shape"
			print self.delta.get_shape().as_list()
			self.clipped_delta	= tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')
			self.global_step	= tf.Variable(0, trainable=False)
			self.loss			= tf.reduce_mean(tf.square(self.clipped_delta), name='loss') # square loss....
			for weight in self.q_w.values():
				self.loss += 1e-4*tf.nn.l2_loss(weight)
			self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
					tf.train.exponential_decay( 	self.learning_rate, self.global_step, 
													self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
			self.optim = tf.train.AdamOptimizer( self.learning_rate_op ).minimize(self.loss, var_list=self.q_w.values(), global_step=self.global_step)

		# setup copy action
		with tf.variable_scope('pred_to_target'):
			self.target_q_w_input = {}
			self.target_q_w_assign_op = {}
			for name in self.q_w.keys():
				self.target_q_w_input[name] = tf.placeholder('float32', self.target_q_w[name].get_shape().as_list(), name=name)
				self.target_q_w_assign_op[name] = self.target_q_w[name].assign(self.target_q_w_input[name])
		
		#tf.initialize_variables(self.q_w.values())
		#tf.initialize_variables(self.target_q_w.values())
		tf.initialize_all_variables().run()
		self.saver = tf.train.Saver(self.q_w.values() + [self.step_op], max_to_keep=30)
		#if model_path:
		# restore the model

	def init_preprocessor(self):
		""" load a pre-processor to convert raw input into state (feature vector?) """
		self.prep = DeepFeatureNetwork([self.batch_size, 224, 224, 3], model_vgg, self.deep_feature_model_path)
		self.prep.init_model()

	def predict(self, state, is_training=True, use_target_q=False):
		if is_training:
			if self.step<10000:
				ep = 1.0
			elif self.step<40000:
				ep = 0.8
			elif self.step<80000:
				ep = 0.6
			elif self.step<160000:
				ep=0.4
			elif self.step<320000:
				ep=0.2
			else:
				ep=0.1
		if use_target_q:
			if self.use_batch_norm:
				q, q_actions = sess.run([self.target_q, self.target_q_action], feed_dict={self.target_s_t: state, self.phase_train:is_training})
			else:
				q, q_actions = sess.run([self.target_q, self.target_q_action], feed_dict={self.target_s_t: state})
		else:
			if self.use_batch_norm:
				q, q_actions = sess.run([self.q, self.q_action], feed_dict={self.s_t: state, self.phase_train:is_training})
			else:
				q, q_actions = sess.run([self.q, self.q_action], feed_dict={self.s_t: state})

		if self.finetune:
			ep=0.1
		qs = []
		actions = []

		for i in range(self.batch_size):
			if is_training and random.random() < ep:
				action_idx = random.randrange(self.action_size)
				actions.append(action_idx)
				qs.append(q[i])
			else:
				actions.append(q_actions[i])
				qs.append(q[i])
				#sorted_q = q.argsort()[i]
		return actions, np.array(qs)


	def update_target_q_network(self):
		for name in self.q_w.keys():
			self.target_q_w_assign_op[name].eval({self.target_q_w_input[name]: self.q_w[name].eval()})

	def get_hist(self, image_data):
		color_hist = []
		for i in range(image_data.shape[0]):
			if self.color_type == 'rgbl':
				color_hist.append(rgbl_hist(image_data[i]))
			elif self.color_type == 'lab':
				color_hist.append(lab_hist(image_data[i]))
			elif self.color_type == 'lab_8k':
				color_hist.append(lab_hist_8k(image_data[i]))
			elif self.color_type == 'tiny':
				color_hist.append(tiny_hist(image_data[i]))
			elif self.color_type == 'tiny28':
				color_hist.append(tiny_hist_28(image_data[i]))
			elif self.color_type == 'VladB':
				color_hist.append(get_global_feature(image_data[i]))

		return np.stack(np.array(color_hist))
		

	def get_state(self, raw_data, target_data, history=None):
		""" get state from the raw input data """
		scores = []
		for i, path in enumerate(target_data):
			#mse = (( target_data[i]-raw_data[i] )**2).mean()*100
			#MSE in Lab color space
			target_lab = color.rgb2lab(target_data[i]+0.5)
			data_lab = color.rgb2lab(raw_data[i]+0.5)
			mse = np.sqrt(np.sum(( target_lab - data_lab )**2, axis=2)).mean()/10.0
			scores.append([-mse])


		if self.use_deep_feature:
			if self.use_color_feature:
				color_features = self.get_hist(raw_data)
				features=[]
				deep_features = self.prep.get_feature(raw_data)
				for i in range(self.batch_size):
					if self.use_history:
						features.append(np.concatenate((deep_features[i], color_features[i], history[i]), axis=0))
					else:
						features.append(np.concatenate((deep_features[i], color_features[i]), axis=0))
				return np.stack(np.array(features)), np.array(scores)
			else:
				features=[]
				deep_features = self.prep.get_feature(raw_data)
				return deep_features, np.array(scores)
		else:
			color_features = self.get_hist(raw_data)
			return np.stack(np.array(color_features)), np.array(scores)

	
	def get_new_state(self, is_training=True, in_order=False, idx=-1, get_raw_images=False):
		""" start a new episode """
		# load a new episode (image?)
		if in_order:
			state_raw, target_state_raw, op_str, fn, raw_images = self._load_images(idx,is_training,in_order=in_order, get_raw_images=get_raw_images)
		else:
			state_raw, target_state_raw, op_str, fn, raw_images = self._load_images(idx,is_training,get_raw_images=get_raw_images)# load image, batch
		history = None
		if self.use_history:
			history = np.zeros([self.batch_size, self.seq_len*action_size])
			state, score = self.get_state(state_raw, target_state_raw, history=history)
		else:
			state, score = self.get_state(state_raw, target_state_raw)
		return state, state_raw, score, target_state_raw, op_str, fn, raw_images, history
	def _load_images(self, offset, is_training,in_order=False,get_raw_images=False):
		if is_training:
			offset = random.randint(0, self.train_img_count-self.batch_size-1)
			img_list = self.train_img_list[offset:offset+self.batch_size]
		else:
			"""
			if not in_order:
				shuffle(self.test_img_list)
			"""
			if offset>=0:
				offset = offset*self.batch_size
			else:
				offset = random.randint(0, self.test_img_count-self.batch_size)
			if offset+self.batch_size > len(self.test_img_list):
				offset = len(self.test_img_list)-self.batch_size
			img_list = self.test_img_list[offset:offset+self.batch_size]
		imgs = []
		target_imgs = []
		op_str = []
		raw_imgs = []
		raw_imgs_raw = []
		raw_imgs_target = []
		for img_path in img_list:
			imgs.append(imresize(imread(img_path, mode='RGB'), (224,224))/255.0-0.5)
			target_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(img_path)),"target"), os.path.basename(img_path).split("__")[0])
			if "__" in os.path.basename(img_path):
				op_str.append(os.path.basename(img_path).split("__")[1])
			else:
				op_str.append("_")
			target_imgs.append(imresize(imread(target_path, mode='RGB'), (224,224))/255.0-0.5)

			if get_raw_images:
				raw_imgs_raw.append(imread(img_path, mode='RGB')/255.0-0.5)
				raw_imgs_target.append(imread(target_path, mode='RGB')/255.0-0.5)
		if len(raw_imgs_raw)>0:
			for raw in raw_imgs_raw:
				if self.logging:
					print raw.shape
			raw_imgs.append(raw_imgs_raw)
		if len(raw_imgs_target)>0:
			raw_imgs.append(raw_imgs_target)
		fns = [os.path.basename(path) for path in img_list]
		return np.array(np.stack(imgs, axis=0)), np.array(np.stack(target_imgs, axis=0)), op_str, fns, raw_imgs
		

	def act(self, actions, state_raw, target_state_raw, prev_score, is_training=True, step_count=0, history=None):
		images_after = []
		for i, action_idx in enumerate(actions):
			# apply action to image raw
			if action_idx==-1:
				images_after.append(state_raw[i])
			else:
				#images_after.append(take_action(state_raw[i], action_idx, sess))
				images_after.append(take_action(state_raw[i], action_idx))
		
		images_after_np = np.stack(images_after, axis=0)
		
		if self.use_history:
			for idx in range(self.batch_size):
				history[idx][step_count*action_size+actions[idx]] = 1
			new_state, new_score = self.get_state(images_after_np, target_state_raw, history=history)
		else:
			new_state, new_score = self.get_state(images_after_np, target_state_raw)

		if not is_training:
			return new_state,images_after_np,None,None,history

		reward = new_score - prev_score
		return new_state, images_after_np, reward, new_score, history

	def act_on_raw_img(self, actions, state_raw):
		images_after = []
		for i, action_idx in enumerate(actions):
			# apply action to image raw
			if action_idx==-1:
				images_after.append(state_raw[i])
			else:
				images_after.append(take_action(state_raw[i], action_idx))
		return images_after

	def test(self, is_batch_test=False, use_target_q=False, in_order=False, idx=-1):
		if is_batch_test:
			test_result_dir = "test/"+self.prefix+"/batch_test_%s" % self.prefix
		else:
			test_result_dir = "test/"+self.prefix+"/step_%010d" % self.step
		if not os.path.exists(test_result_dir):
			os.mkdir(test_result_dir)
		state, state_raw, score_initial, target_state_raw,op_strs, fn, raw_images, history = self.get_new_state(is_training=False, in_order=in_order, idx=idx, get_raw_images=True)
		score = score_initial.copy()
		state_raw_init = state_raw.copy()
		raw_images_raw = raw_images[0]
		raw_images_target = raw_images[1]
		retouched_raw_images = [item.copy() for item in raw_images_raw]
		actions = []
		for i in range(self.seq_len):
			action, q_val = self.predict(state, is_training=False, use_target_q=use_target_q)
			if self.logging:
				print "q_val", q_val
			all_stop = True
			for j in range(self.batch_size):
				if q_val[j][action[j]] <= 0:
					all_stop = all_stop and True
					action[j]=-1
				else:
					all_stop = all_stop and False
			if all_stop:
				print "all negative, stop"
				break
			next_state, next_state_raw, r, new_score, history = self.act(action, state_raw, target_state_raw, score, is_training=False, history=history)
			#next_state, next_state_raw, reward, new_score = self.act(action, state_raw, target_state_raw, score)
			
			state = next_state
			state_raw = next_state_raw
			if self.logging:
				print ""
				print "step", i
				print q_val
				print q_val.argsort()
			actions.append(action)
			score = new_score

			# to save raw image.. in raw resolution...
			if self.save_raw:
				retouched_raw_images = self.act_on_raw_img(action, retouched_raw_images)

		if self.logging:
			print actions

		state, final_score = self.get_state(state_raw, target_state_raw, history=history)
		score_diff = final_score - score_initial
		if self.logging:
			print ""
			print "final_score:\t", final_score
			print "original_score:\t", score_initial

		raw_dir = os.path.join(test_result_dir, 'raw')
		if not os.path.exists(raw_dir):
			os.mkdir(raw_dir)

		initial_score = 0 
		retouched_score = 0
		for i in range(state_raw.shape[0]):
			try:
				actions_str = []
				for j in range(len(actions)):
					if actions[j][i]!=-1:
						actions_str.append(str(actions[j][i]))
				actions_desc = "_".join(actions_str)
				if idx == -1:
					random_id = random.randrange(999999999)
				else:
					random_id = idx
				with open(os.path.join(test_result_dir, '%s.log'%fn[i]), 'wb') as f:
					f.write("%s_%f_%s.png\n" % (fn[i], score_diff[i][0], actions_desc))
				image = Image.fromarray(np.uint8(np.clip((state_raw[i]+0.5)*255, 0, 255)))
				image.save(os.path.join(test_result_dir, "%s_%f_retouched.png" % (fn[i], score_diff[i][0])))
                
				image_target = Image.fromarray(np.uint8(np.clip((target_state_raw[i]+0.5)*255, 0, 255)))
				image_target.save(os.path.join(test_result_dir, "%s_target.png" % (fn[i])))
				
				image_init = Image.fromarray(np.uint8(np.clip((state_raw_init[i]+0.5)*255, 0, 255)))
				image_init.save(os.path.join(test_result_dir, "%s_raw_%s.png" % (fn[i], op_strs[i])))
                
				raw_dir_dir = os.path.join(raw_dir, fn[i].split("__")[0])
                
				if not os.path.exists(raw_dir_dir):
					os.mkdir(raw_dir_dir)
                
				if self.save_raw:
					raw_image_raw		= Image.fromarray(np.uint8(np.clip((raw_images_raw[i]+0.5)*255,0,255)))
					raw_image_retouched	= Image.fromarray(np.uint8(np.clip((retouched_raw_images[i]+0.5)*255,0,255)))
					raw_image_target	= Image.fromarray(np.uint8(np.clip((raw_images_target[i]+0.5)*255,0,255)))
                    
					target_lab 		= color.rgb2lab(raw_images_target[i] + 0.5)
					retouched_lab 	= color.rgb2lab(retouched_raw_images[i] + 0.5)
					initial_lab 	= color.rgb2lab(raw_images_raw[i] + 0.5)
					initial_score	+= - np.sqrt(np.sum(( target_lab - initial_lab )**2, axis=2)).mean()/10.0
					retouched_score	+= - np.sqrt(np.sum(( target_lab - retouched_lab )**2, axis=2)).mean()/10.0
					try:
						raw_image_raw.save(os.path.join(raw_dir_dir, "%s_raw.png" %(fn[i])))
						raw_image_retouched.save(os.path.join(raw_dir_dir, "%s_retouched_%f_%s.png" %(fn[i], score_diff[i][0], actions_desc)))
						raw_image_target.save(os.path.join(raw_dir_dir, "%s_target.png" %(fn[i])))
					except Exception as e:
						print "exception occurred"
						print str(e)
				else:
					target_lab 		= color.rgb2lab(target_state_raw[i] + 0.5)
					retouched_lab 	= color.rgb2lab(state_raw[i] + 0.5)
					initial_lab 	= color.rgb2lab(state_raw_init[i] + 0.5)
					initial_score	+= - np.sqrt(np.sum(( target_lab - initial_lab )**2, axis=2)).mean()/10.0
					retouched_score	+= - np.sqrt(np.sum(( target_lab - retouched_lab )**2, axis=2)).mean()/10.0
			except Exception as e:
				print str(e)
		return initial_score, retouched_score

class DeepFeatureNetwork:
	def __init__(self, input_size, model_func, model_path):
		self.input_tensor = tf.placeholder(tf.float32, shape=input_size)
		self.feature, self.weights = model_func(self.input_tensor, model_path)
		
	def init_model(self):
		tf.initialize_variables(self.weights).run()

	def get_feature(self, in_data):
		return sess.run(self.feature, feed_dict={self.input_tensor: in_data+0.5})

def enqueue_samples(self, coord):
	repeat = 0
	state, state_raw, score, target_state_raw, _, _, _, history= self.get_new_state()

	step_count = 0

	#for step in range(self.max_step):
	while not coord.should_stop():
		if repeat>self.seq_len:
			repeat = 0
			state, state_raw, score, target_state_raw, _, _, _, history = self.get_new_state()
			terminal = False
			step_count = 0
			if self.use_history:
				del history
				history = np.zeros([self.batch_size, self.seq_len * action_size])

		# 1. predict
		action, q_val = self.predict(state)

		step_count += 1

		# 2. act
		if self.use_history:
			next_state, next_state_raw, reward, new_score, history = self.act(action, state_raw, target_state_raw, score, history=history)
		else:
			next_state, next_state_raw, reward, new_score, _ = self.act(action, state_raw, target_state_raw, score)


		terminal = reward<=0
		if np.sum(terminal)>self.batch_size/2:
			repeat+=5
		else:
			repeat+=1
		terminal= np.reshape(terminal, (self.batch_size, -1))
		action	= np.reshape(action, (self.batch_size, -1))
		
		for i in range(self.enqueue_repeat):
			sess.run(self.enqueue_op, feed_dict={self.s_t_single: state, self.s_t_plus_1_single: next_state, self.reward_single: reward, self.action_single: action, self.terminal_single: terminal})

		state_raw = next_state_raw
		state = next_state
		score = new_score

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path")
	parser.add_argument("--prefix")
	args = parser.parse_args()
	model_path = args.model_path
	prefix = args.prefix
	
	if prefix == None:
		print "please provide a valid prefix"
		sys.exit(1)
	elif os.path.exists('./test/'+prefix):
		print "duplicated prefix"
		sys.exit(1)
	else:
		os.makedirs('./test/'+prefix)

	agent = Agent(prefix)
	config = tf.ConfigProto()

	with tf.Session(config=config) as sess:
		agent.init_preprocessor()
		agent.init_model()
		agent.init_img_list()
		if model_path:
			agent.load_model(model_path)
			print ("run test with model {}".format(model_path))
			agent.step = 0
			for i in range(agent.test_count):
				init_score, final_score = agent.test(in_order=True, idx=i)
		else:
			print ("start training with prefix {}".format(prefix))
			agent.train_with_queue()
