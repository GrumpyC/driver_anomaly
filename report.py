import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

result = np.load('report.npy',allow_pickle=True,)


result = np.array(result)
fig, ax = plt.subplots()
plt.axhline(0.001, 0, 1, color='yellow', linestyle = 'dashdot' , label='Program threshold as warning')
plt.axvline(68000, 0, 1, color='yellow', linestyle = 'dotted',   label='Annotated as warning')
plt.axvline(72000, 0, 1, color='yellow', linestyle = 'dotted')
plt.axvline(100000, 0, 1, color='yellow', linestyle = 'dotted')
plt.axvline(104000, 0, 1, color='yellow', linestyle = 'dotted')

plt.axhline(0.01, 0, 1, color='orange', linestyle = 'dashdot' , label='Program threshold as serious')
plt.axvline(136000, 0, 1, color='orange', linestyle = 'dotted', label='Annotated as serious')
plt.axvline(139000, 0, 1, color='orange', linestyle = 'dotted')
plt.axvline(33000, 0, 1, color='orange', linestyle = 'dotted')
plt.axvline(39000, 0, 1, color='orange', linestyle = 'dotted')

plt.axhline(0.02, 0, 1, color='red', linestyle = 'dashdot',   label='Program threshold as critical')
plt.axvline(153000, 0, 1, color='red', linestyle = 'dotted',  label='Annotated as critical')
plt.axvline(159900, 0, 1, color='red', linestyle = 'dotted')
plt.axvline(167000, 0, 1, color='red', linestyle = 'dotted')
plt.axvline(175000, 0, 1, color='red', linestyle = 'dotted')



legend = ax.legend(loc='upper left', shadow=True, fontsize='medium')

ax.plot(result[:,1],result[:,0])
ax.set_facecolor('xkcd:light grey')

ax.set(xlabel='time (s)', ylabel='MSE',
       title='Error rate to time elapsed')
ax.grid()
fig.patch.set_facecolor('xkcd:light grey')
fig.savefig("report_plot.png")
plt.show()