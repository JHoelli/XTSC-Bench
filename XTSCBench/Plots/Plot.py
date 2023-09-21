import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("tkagg")
import seaborn as sns 
def plot_one_example(original,exp, value=None):
    item=original
    fig, axn = plt.subplots(
                len(item[0]), 1, sharex=True, sharey=True
            )
            # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    axn012 = axn.twinx()
    sns.heatmap(
                exp[0].reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axn,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
    sns.lineplot(
                x=range(0, len(item[0][0].reshape(-1))),
                y=item[0][0].flatten(),
                ax=axn012,
                color="white",

            )
    plt.title(value)
    plt.show()

def plot_one_example_with_meta(original,exp,reference_sample, value=None):
    item=original
    fig, axs = plt.subplots(2, 1,sharex=True)
            # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    axn012 = axs[0].twinx()
    axn012.set(xticklabels=[])
    axn012.set(yticklabels=[])
    axs[0].set(yticklabels=[])
    sns.heatmap(
                exp[0].reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axs[0],
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
    sns.lineplot(
                x=range(0, len(item[0][0].reshape(-1))),
                y=item[0][0].flatten(),
                ax=axn012,
                color="white",

            )
    sns.heatmap(
                reference_sample.reshape(1, -1),
                fmt="g",
                cbar=False,
                cmap="viridis",
                ax=axs[-1],
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
    axs[0].set(ylabel='E(x)')
    axs[0].set(xticklabels=[])
    axs[0].set(yticklabels=[])
    axs[-1].set(ylabel='GT')
    axs[-1].set(xticklabels=[])
    axs[-1].set(yticklabels=[])

    plt.legend([],[], frameon=False)
    plt.title(value)
    plt.show()