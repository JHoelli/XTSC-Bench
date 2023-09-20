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
