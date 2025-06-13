import matplotlib.pyplot as plt

def plot_deep_ensembles():
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, color in enumerate(["skyblue", "lightgreen", "lightcoral"]):
        ax.plot(
            [0, 1, 2], [0.5 + 0.3 * i, 0.6 + 0.2 * i, 0.7 + 0.1 * i],
            label=f"Model {i+1}", lw=3, color=color
        )
    ax.scatter(3, 0.65, label="Final Prediction", color="gold", s=200)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Input", "Hidden", "Hidden", "Output"])
    ax.set_title("Deep Ensembles: Combining Predictions", fontsize=16)
    ax.legend(fontsize=12)
    plt.show()

plot_deep_ensembles()
