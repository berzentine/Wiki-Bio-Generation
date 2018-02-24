import matplotlib
matplotlib.use('agg')
import seaborn as sns
import pandas as pd


def plot(train_losses, val_losses, save):
    """ Plots the loss curves """
    #print(train_losses, val_losses)
    df = pd.DataFrame({"Train":train_losses, "Val":val_losses},
                      columns=["Train","Val"])
    df["Epochs"] = df.index
    var_name = "Loss Type"
    value_name = "Loss"
    df = pd.melt(df, id_vars=["Epochs"], value_vars=["Train", "Val"],
                 var_name=var_name, value_name=value_name)
    sns.tsplot(df, time="Epochs", unit=var_name, condition=var_name,
               value=value_name)
    matplotlib.pyplot.savefig(save, bbox_inches="tight")
