import json
from matplotlib import pyplot as plt
from matplotlib import cm

def load_result_dict():
    d = json.load(open('wer_results.json'))
    return d

def plot(wers, color = None, label = None):
    plt.ion()
    x = [line[1] for line in wers]
    y = [line[0] for line in wers]
    plt.plot(x,y, color = color, label = label)
    return x, y

def plot_orthographic(d = None, condition= 'orthographic'):
    if not d: d = load_result_dict()
    # colors = ['red','blue','green','purple','orange','black']
    colors = cm.get_cmap('tab20').colors
    i = 0
    plt.ion()
    plt.figure()
    for label, value in d.items():
        wers = value[condition]
        if not wers: continue
        color = colors[i]
        i += 1
        label = str(label.strip('/'))
        print([label,color])
        plot(wers,color,label)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for line in legend.get_lines():
        line.set_linewidth(6)
    plt.grid(alpha = .5)
    plt.ylabel('wer')
    plt.xlabel('training steps')
    plt.title(condition)
