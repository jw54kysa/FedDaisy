import matplotlib.pyplot as plt
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