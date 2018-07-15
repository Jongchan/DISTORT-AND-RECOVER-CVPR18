import numpy as np

f = open('./test/regularizer_2/test.txt', 'rb')
#f = open('./test/normalized_nonlinear_0/test.txt', 'rb')
#f = open('./test/pretrained_0/test.txt', 'rb')
#f = open('./test/regularizer_1/test.txt', 'rb')
count=63
#f = open('./test/shutterstock_2/test.txt', 'rb')
#count=250


lines = f.readlines()
f.close()
print len(lines)
for i in range(len(lines)//3):
	step_line	= lines[3*i]
	raw_line	= lines[3*i+1].split(" ")
	final_line	= lines[3*i+2].split(" ")
	step = step_line.split(" ")[-1]
	raw_floats = [float(item) for item in raw_line]
	final_floats = [float(item) for item in final_line]

	raw_np = np.array(raw_floats[:count])
	final_np = np.array(final_floats[:count])
	diff_np = final_np - raw_np
	raw_sum = np.sum(raw_np)
	final_sum = np.sum(final_np)
	diff_sum = np.sum(diff_np)
	print step_line
	print "raw_sum", raw_sum
	print "final_sum", final_sum
	print "diff_sum", diff_sum
	#print "diff_mean", diff_sum/len(final_line)*10/4
	#print "score mean", abs(final_sum/len(final_line))*10/4
	print "diff_mean", diff_sum/raw_np.size*10/4
	print "score mean", abs(final_sum/raw_np.size)*10/4
