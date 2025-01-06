import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle

def createLossAccPlot(exp_path):
    # Load the data
    with open(exp_path+"/testLosses.pck", "rb") as f:
        testLosses = pickle.load(f)

    with open(exp_path+"/trainLosses.pck", "rb") as f:
        trainLosses = pickle.load(f)

    with open(exp_path+"/testACCs.pck", "rb") as f:
        testAcc = pickle.load(f)

    with open(exp_path+"/trainACCs.pck", "rb") as f:
        trainAcc = pickle.load(f)

    def calculate_column_averages(data):
        columns = zip(*data)
        averages = [sum(column) / len(column) for column in columns]
        return averages

    # Get averages
    testLossAvg = calculate_column_averages(testLosses)
    trainLossAvg = calculate_column_averages(trainLosses)
    testAccAvg = calculate_column_averages(testAcc)
    trainAccAvg = calculate_column_averages(trainAcc)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # loss
    ax1.plot(testLossAvg, linestyle='-', color='b', label='Test Loss')
    ax1.plot(trainLossAvg, linestyle='--', color='b', label='Train Loss')
    ax1.set_title("Experiment Accuracy & Losses", fontsize=10)
    ax1.set_xlabel("Report Round", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10, color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # accuracy
    ax2 = ax1.twinx()
    ax2.plot(testAccAvg, linestyle='-', color='g', label='Test Accuracy')
    ax2.plot(trainAccAvg, linestyle='--', color='g', label='Train Accuracy')
    ax2.set_ylabel("Accuracy", fontsize=10, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1), bbox_transform=ax1.transAxes, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(exp_path+"/lossAcc.png")

# Plot random size client samples
def plot_rss(client_idxs, visits, path):
    counts = []
    for l in client_idxs:
        counts.append(len(l))
    # counts.sort()

    combined = sorted(zip(counts, visits), key=lambda x: x[0])

    # Separate the sorted lists
    sorted_counts, sorted_visits = zip(*combined)

    # Convert back to lists
    counts = list(sorted_counts)
    visits = list(sorted_visits)

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Bar plot for sample size
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.bar(np.arange(len(counts)), counts, label='Sample Size', alpha=0.7, color='blue')
    ax1.set_title("Client Sample Size")
    ax1.set_xlabel("Client")
    ax1.set_ylabel("Sample Size", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for visits
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(visits)), visits, label='Visits', color='red', marker='o')
    ax2.set_ylabel("Visits", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Adding legends for both plots
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Save and display the plot
    plt.savefig(path)
    plt.show()
