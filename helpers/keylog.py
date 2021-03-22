import matplotlib.pyplot as plt
import numpy as np

def plotDDT(keystroke_data):
    ddt_times = [float(x['keydelay']) for x in keystroke_data]
    plotGraph(ddt_times, "DDT")

def plotDUT(keystroke_data):
    ddu_times = [float(x['keyhold']) for x in keystroke_data]
    plotGraph(ddu_times, "DUT")

def plotGraph(y, event_type):
    data = y
    x = list(range(len(data)))

    # Average
    average = np.mean(data)
    # Words Per Minute = (Chr / 5) / Time
    wpm = len(data) / 5

    # MatPlotLib Handling
    plt.title("Time Elapsed for %s Events" % event_type)
    plt.ylabel("Key Number")
    plt.ylabel("Milliseconds")
    plt.plot(x, y)
    # Format average display box
    plt.text(5, 35, ("WPM: ", wpm, "Average", average), style='italic',
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()
