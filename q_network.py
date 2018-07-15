import tensorflow as tf
import numpy as np
def q_network(input_tensor, scope_name, input_length=512, num_action=11, duel=False, batch_norm=False, phase_train=None):
	with tf.variable_scope(scope_name):

		if duel:
			# value
			weights = {}
			if batch_norm:
				beta	= tf.Variable(tf.constant(0.0, shape=[input_length], name='beta', trainable=True))
				gamma	= tf.Variable(tf.constant(1.0, shape=[input_length], name='gamma', trainable=True))
				batch_mean, batch_var = tf.nn.moments(input_tensor, [0,1,2], name='moments')
				ema = tf.train.ExponentialMovingAverage(decay=0.5)
				def mean_var_with_update():
					ema_apply_op = ema.apply([batch_mean, batch_var])
					with tf.control_dependencies([ema_apply_op]):
						return tf.identity(batch_mean), tf.identity(batch_var)
				mean, var = tf.cond(phase_train, mean_var_with_updates, lambda: (ema.average(batch_mean), ema.average(batch_var)))
				input_tensor = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, 1e-3)
				weights['mean']	= mean
				weights['var']	= var
				weights['beta']	= beta
				weights['gamma']= gamma

			v_fc1_weights = tf.get_variable("v_fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc1_biases = tf.get_variable("v_fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			v_fc2_weights	= tf.get_variable("v_fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc2_biases	= tf.get_variable("v_fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			v_fc3_weights	= tf.get_variable("v_fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc3_biases	= tf.get_variable("v_fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			v_fc4_weights	= tf.get_variable("v_fc4_weights", [512, 1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			v_fc4_biases	= tf.get_variable("v_fc4_biases", [1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0)))
    
			weights['v_fc1_weights']	= v_fc1_weights
			weights['v_fc1_biases']		= v_fc1_biases
			weights['v_fc2_weights']	= v_fc2_weights
			weights['v_fc2_biases']		= v_fc2_biases
			weights['v_fc3_weights']	= v_fc3_weights
			weights['v_fc3_biases']		= v_fc3_biases
			weights['v_fc4_weights']	= v_fc4_weights
			weights['v_fc4_biases']		= v_fc4_biases
    
			v_tensor = tf.nn.dropout(input_tensor, 1.0)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc1_weights)+v_fc1_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc2_weights)+v_fc2_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc3_weights)+v_fc3_biases)
			v_tensor = tf.matmul(v_tensor, v_fc4_weights)+v_fc4_biases


			# duel
			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			fc2_weights	= tf.get_variable("fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc2_biases	= tf.get_variable("fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc4_biases	= tf.get_variable("fc4_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc2_weights']	= fc2_weights
			weights['fc2_biases']	= fc2_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights)+fc2_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.matmul(tensor, fc4_weights)+fc4_biases
			print weights
			q = v_tensor + (tensor - tf.reduce_mean(tensor, reduction_indices=1, keep_dims=True))
			return q, weights
		else:

			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			fc2_weights	= tf.get_variable("fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc2_biases	= tf.get_variable("fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc4_biases	= tf.get_variable("fc4_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights = {}
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc2_weights']	= fc2_weights
			weights['fc2_biases']	= fc2_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights)+fc2_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.matmul(tensor, fc4_weights)+fc4_biases
    
			return tensor, weights

def q_network_shallow(input_tensor, scope_name, input_length=512, num_action=11, duel=False):
	with tf.variable_scope(scope_name):

		if duel:
			# value
			v_fc1_weights = tf.get_variable("v_fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc1_biases = tf.get_variable("v_fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
    
			v_fc3_weights	= tf.get_variable("v_fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc3_biases	= tf.get_variable("v_fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			v_fc4_weights	= tf.get_variable("v_fc4_weights", [512, 1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			v_fc4_biases	= tf.get_variable("v_fc4_biases", [1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0)))
    
			weights = {}
			weights['v_fc1_weights']	= v_fc1_weights
			weights['v_fc1_biases']		= v_fc1_biases
			weights['v_fc3_weights']	= v_fc3_weights
			weights['v_fc3_biases']		= v_fc3_biases
			weights['v_fc4_weights']	= v_fc4_weights
			weights['v_fc4_biases']		= v_fc4_biases
    
			v_tensor = tf.nn.dropout(input_tensor, 1.0)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc1_weights)+v_fc1_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc3_weights)+v_fc3_biases)
			v_tensor = tf.matmul(v_tensor, v_fc4_weights)+v_fc4_biases


			# duel
			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc4_biases	= tf.get_variable("fc4_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.matmul(tensor, fc4_weights)+fc4_biases
			print weights
			q = v_tensor + (tensor - tf.reduce_mean(tensor, reduction_indices=1, keep_dims=True))
			return q, weights
		else:

			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			fc2_weights	= tf.get_variable("fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc2_biases	= tf.get_variable("fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc4_biases	= tf.get_variable("fc4_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights = {}
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc2_weights']	= fc2_weights
			weights['fc2_biases']	= fc2_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights)+fc2_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.matmul(tensor, fc4_weights)+fc4_biases
    
			return tensor, weights
def q_network_large(input_tensor, scope_name, input_length=512, num_action=11, duel=False):
	with tf.variable_scope(scope_name):

		if duel:
			# value
			v_fc1_weights = tf.get_variable("v_fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc1_biases = tf.get_variable("v_fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			v_fc2_weights	= tf.get_variable("v_fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc2_biases	= tf.get_variable("v_fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			v_fc3_weights	= tf.get_variable("v_fc3_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc3_biases	= tf.get_variable("v_fc3_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			v_fc4_weights	= tf.get_variable("v_fc4_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			v_fc4_biases	= tf.get_variable("v_fc4_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			v_fc5_weights	= tf.get_variable("v_fc5_weights", [512, 1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			v_fc5_biases	= tf.get_variable("v_fc5_biases", [1], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0)))
    
			weights = {}
			weights['v_fc1_weights']	= v_fc1_weights
			weights['v_fc1_biases']		= v_fc1_biases
			weights['v_fc2_weights']	= v_fc2_weights
			weights['v_fc2_biases']		= v_fc2_biases
			weights['v_fc3_weights']	= v_fc3_weights
			weights['v_fc3_biases']		= v_fc3_biases
			weights['v_fc4_weights']	= v_fc4_weights
			weights['v_fc4_biases']		= v_fc4_biases
			weights['v_fc5_weights']	= v_fc5_weights
			weights['v_fc5_biases']		= v_fc5_biases
    
			v_tensor = tf.nn.dropout(input_tensor, 1.0)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc1_weights)+v_fc1_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc2_weights)+v_fc2_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc3_weights)+v_fc3_biases)
			v_tensor = tf.nn.relu(tf.matmul(v_tensor, v_fc4_weights)+v_fc4_biases)
			v_tensor = tf.matmul(v_tensor, v_fc5_weights)+v_fc5_biases


			# duel
			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			fc2_weights	= tf.get_variable("fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc2_biases	= tf.get_variable("fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc4_biases	= tf.get_variable("fc4_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc5_weights	= tf.get_variable("fc5_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc5_biases	= tf.get_variable("fc5_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc2_weights']	= fc2_weights
			weights['fc2_biases']	= fc2_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
			weights['fc5_weights']	= fc5_weights
			weights['fc5_biases']	= fc5_biases
    
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights)+fc2_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc4_weights)+fc4_biases)
			tensor = tf.matmul(tensor, fc5_weights)+fc5_biases
			print weights
			q = v_tensor + (tensor - tf.reduce_mean(tensor, reduction_indices=1, keep_dims=True))
			return q, weights
		else:

			fc1_weights = tf.get_variable("fc1_weights", [input_length, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc1_biases = tf.get_variable("fc1_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
		
			fc2_weights	= tf.get_variable("fc2_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc2_biases	= tf.get_variable("fc2_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc3_weights	= tf.get_variable("fc3_weights", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc3_biases	= tf.get_variable("fc3_biases", [4096], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
    
			fc4_weights	= tf.get_variable("fc4_weights", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/4096)))
			fc4_biases	= tf.get_variable("fc4_biases", [512], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
    
			fc5_weights	= tf.get_variable("fc5_weights", [512, num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/512)))
			fc5_biases	= tf.get_variable("fc5_biases", [num_action], initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0/num_action)))
    
			weights = {}
			weights['fc1_weights']	= fc1_weights
			weights['fc1_biases']	= fc1_biases
			weights['fc2_weights']	= fc2_weights
			weights['fc2_biases']	= fc2_biases
			weights['fc3_weights']	= fc3_weights
			weights['fc3_biases']	= fc3_biases
			weights['fc4_weights']	= fc4_weights
			weights['fc4_biases']	= fc4_biases
			weights['fc5_weights']	= fc5_weights
			weights['fc5_biases']	= fc5_biases
    
			tensor = tf.nn.dropout(input_tensor, 1.0)
			tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights)+fc1_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights)+fc2_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights)+fc3_biases)
			tensor = tf.nn.relu(tf.matmul(tensor, fc4_weights)+fc4_biases)
			tensor = tf.matmul(tensor, fc5_weights)+fc5_biases
    
			return tensor, weights
