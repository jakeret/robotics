'''
Data set of soft pneumatic arm
'''
# --------------------------------------------------------------------------- #
# import
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------- #
# fixed parameter
input_size = 6 # {pASP, pBSP, DC1, DC2, DC3, DC4}
output_size = 3 # {alpha, pA, pB}
# --------------------------------------------------------------------------- #
# chose data for train/valid/test
filepath_train = ("Matthias_train.txt")
filepath_test = ("Matthias_test.txt")

# load data (not normalized)
raw_train_data = np.loadtxt(filepath_train)
N_train = np.shape(raw_train_data)[0]

x_train_raw = np.atleast_2d(raw_train_data[:,0:input_size])
y_train_raw = np.atleast_2d(raw_train_data[:,input_size:])
# --------------------------------------------------------------------------- #
# Plotting 
plt.close("all")

fig1, (ax1, ax2, ax3, ax4)= plt.subplots(nrows=4, ncols=1, sharex='all')
ax1.plot(x_train_raw[:,0],'b', label="pASP")
ax1.plot(x_train_raw[:,1],'r', label="pBSP")
ax1.set_ylabel('pressure setpoints [-]')
ax1.grid(color='k', linestyle='-', linewidth=0.1)
ax1.legend(loc=1)
#----- #
ax2.plot(x_train_raw[:,2],'b', label="DC1")
ax2.plot(x_train_raw[:,3],'r', label="DC2")
ax2.plot(x_train_raw[:,4],'g', label="DC3")
ax2.plot(x_train_raw[:,5],'k', label="DC4")
ax2.set_ylabel('DC [-]')
ax2.grid(color='k', linestyle='-', linewidth=0.1)
ax2.legend(loc=1)
#----- #
ax3.plot(y_train_raw[:,0], 'b', label="alpha")
ax3.set_ylabel('angle [-]')
ax3.grid(color='k', linestyle='-', linewidth=0.1)
ax3.legend(loc=1)
#----- #
ax4.plot(y_train_raw[:,1], 'b', label="pA")
ax4.plot(y_train_raw[:,2], 'r', label="pB")
ax4.set_xlabel('time index k [-]')
ax4.set_ylabel('pressure [-]')
ax4.grid(color='k', linestyle='-', linewidth=0.1)
ax4.legend(loc=1)

plt.show()
# --------------------------------------------------------------------------- #