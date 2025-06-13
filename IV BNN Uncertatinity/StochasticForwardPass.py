import matplotlib.pyplot as plt
import networkx as nx

def plot_mc_dropout():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Input", "Hidden1"), ("Input", "Hidden2"), ("Input", "Hidden3"),
        ("Hidden1", "Output"), ("Hidden2", "Output"), ("Hidden3", "Output")
    ])

    pos = {
        "Input": (0, 0),
        "Hidden1": (1, 1), "Hidden2": (1, 0), "Hidden3": (1, -1),
        "Output": (2, 0)
    }

    labels = {
        "Input": "Input",
        "Hidden1": "Hidden (Dropout)", "Hidden2": "Hidden (Dropout)", "Hidden3": "Hidden (Dropout)",
        "Output": "Output"
    }

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        G, pos, with_labels=True, labels=labels, node_color="skyblue",
        node_size=2000, font_size=12, font_color="black", edge_color="gray"
    )
    plt.title("MC Dropout: Stochastic Forward Passes", fontsize=16)
    plt.axis("off")
    plt.show()

plot_mc_dropout()
