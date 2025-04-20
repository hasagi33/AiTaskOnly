import pickle
import matplotlib.pyplot as plt
import os
import matplotlib

backends = ['TkAgg', 'Qt5Agg', 'WXAgg']
for backend in backends:
    try:
        matplotlib.use(backend)
        break
    except:
        continue

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(script_dir, "loss_plot_3.fig.pickle")

if os.path.exists(fig_path):
    with open(fig_path, "rb") as f:
        fig = pickle.load(f)
    plt.show()
else:
    print(f"File not found: {fig_path}")