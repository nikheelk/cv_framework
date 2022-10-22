import matplotlib.pyplot as plt

# Plotting graphs
def plot_single(title, train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    fig.suptitle(title)
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.tight_layout()
    plt.show()

def plot_multi(ls_title, ls_train_losses, ls_train_acc, ls_test_losses, ls_test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    fig.suptitle(" | ".join(ls_title))

    for train_losses in ls_train_losses:
        axs[0, 0].plot(train_losses)
    axs[0, 0].legend(ls_title)
    axs[0, 0].set_title("Training Loss")
  
    for train_acc in ls_train_acc:
        axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].legend(ls_title)
    axs[1, 0].set_title("Training Accuracy")
  
    for test_losses in ls_test_losses:
        axs[0, 1].plot(test_losses)
    axs[0, 1].legend(ls_title)
    axs[0, 1].set_title("Test Loss")
  
    for test_acc in ls_test_acc:
        axs[1, 1].plot(test_acc)
    axs[1, 1].legend(ls_title)
    axs[1, 1].set_title("Test Accuracy")