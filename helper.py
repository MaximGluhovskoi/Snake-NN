from statistics import mean
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, records, mean10):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(mean10)
    plt.plot(mean_scores)
    plt.plot(scores)
    #print(mean10)
    plt.plot(records)
    plt.ylim(ymin=0)

    plt.text(len(records)-1, records[-1], str(records[-1]))
    plt.text(len(mean10)-1, mean10[-1], str(mean10[-1]))
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))