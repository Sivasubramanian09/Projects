# We have graph represented as adjacency list using python dictionary and list.Generate a visual graph from this data 
# (e.g., use dot library/tool or convert this data to json/javascript and use some javascript graph viz library). 
# Generate random graphs for development/testing.

import pydot  # Importing the pydot module for graph creation

# Function to create a graph from an adjacency list
def create_graph(adj_list, graph_type = 'graph'):
    graph = pydot.Dot(graph_type=graph_type, strict=True)         #Creating a Dot object representing the graph
    nodes = list(adj_list.keys())                                 #Extracting the nodes from the adjacency list

    # Adding nodes to the graph
    for node in nodes:
        graph.add_node(pydot.Node(name=node, style="filled", fillcolor="deepskyblue"))                     #Adding each node as a Node object to the graph

    # Adding edges to the graph
    for node, adj_nodes in adj_list.items():
        for adj_node in adj_nodes:
            graph.add_edge(pydot.Edge(node, adj_node))           #Adding an edge between each node and its adjacent nodes

    return graph                                                 #Returning the created graph

# Function to add a node to the adjacency list
def add_node(adj_list, node):
    if node not in adj_list:                                      #checking if the node exist in the adjacency list
        adj_list[node] = []                                       #adding empty list of adjacent nodes

# Function to remove a node from the adjacency list
def remove_node(adj_list, node):
    if node in adj_list:                                     #checking node present in adj_list or not
        del adj_list[node]                                   #if yes, deleting node                            
        for adj_nodes in adj_list.values():                  #running for loop to remove the node values
            # print(adj_nodes)
            if node in adj_nodes:
                adj_nodes.remove(node)                      #Remove the node from the adjacent nodes of all other nodes

# Function to add an edge between two nodes in the adjacency list
def add_edge(adj_list, node1, node2):
    if node1 not in adj_list:                               #checking If node1 exist in adjacency list or not, 
        adj_list[node1] = []                                #if yes,add it with an empty list of adjacent nodes
    if node2 not in adj_list:                               #checking If node2 exist in the adjacency list,
        adj_list[node2] = []                                #if yes,add it with an empty list of adjacent nodes
    adj_list[node1].append(node2)                           #else add node2 to the list of adjacent nodes of node1

# Function to remove an edge between two nodes in the adjacency list
def remove_edge(adj_list, node1, node2):
    if node1 in adj_list and node2 in adj_list[node1]:       #checking If the edge between node1 and node2 exists
        adj_list[node1].remove(node2)                        #if yes, removing the edges

# Example usage:
adj_list = {
    "A": ["C", "E"],
    "B": ["C", "F", "A"],
    "C": ["B", "E"],
    "D": ["F"],
    "E": ["D", "A"],
    "F": ["A"],
}

# add_node(adj_list,"G")
# #add_node(adj_list,"Z")
# add_edge(adj_list,"Z","B")
# add_edge(adj_list,"D","B")
# # add_edge(adj_list,"C","G")
# # # add_edge(adj_list,"A","Z")
# # add_edge(adj_list,"F","G")
# remove_edge(adj_list,"A","C")
# # remove_edge(adj_list,"C","G")
remove_node(adj_list,"F")




graph = create_graph(adj_list, 'digraph')  # Create a graph from the modified adjacency list
graph.write_png("out.png")  # Write the graph to a PNG file

