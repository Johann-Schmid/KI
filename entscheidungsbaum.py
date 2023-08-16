import matplotlib.pyplot as plt
import networkx as nx


# Using a hierarchical layout for better visualization
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root, parsed=parsed)

    return pos

# Create a directed graph
G =nx.DiGraph()

# Nodes for the animal decision tree
G.add_node("Root", text="Hat es Federn?")
G.add_node("A", text="Es ist ein Vogel.")
G.add_node("B", text="Hat es vier Beine?")
G.add_node("C", text="Es ist eine Katze.")
G.add_node("D", text="Lebt es im Wasser?")
G.add_node("E", text="Es ist ein Fisch.")
G.add_node("F", text="Es ist eine Schlange.")

# Edges for the animal decision tree
G.add_edge("Root", "A", text="Ja")
G.add_edge("Root", "B", text="Nein")
G.add_edge("B", "C", text="Ja")
G.add_edge("B", "D", text="Nein")
G.add_edge("D", "E", text="Ja")
G.add_edge("D", "F", text="Nein")

# Draw the tree using the hierarchical layout
hierarchical_positions = hierarchy_pos(G, "Root")
plt.figure(figsize=(12, 8))

# Draw nodes, labels, edges, and edge labels set the color to yellow
nx.draw_networkx_nodes(G, hierarchical_positions, node_size=5000, node_color='orange')
nx.draw_networkx_labels(G, hierarchical_positions, labels=nx.get_node_attributes(G, 'text'))
nx.draw_networkx_edges(G, hierarchical_positions)
nx.draw_networkx_edge_labels(G, hierarchical_positions, edge_labels=nx.get_edge_attributes(G, 'text'))

# Display the plot
plt.title("Entscheidungsbaum für Tiere")
plt.axis('off')
plt.show()
