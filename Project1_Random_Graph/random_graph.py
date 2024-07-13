import pydot  #Importing the pydot module 
import random  #Importing the random module 

def random_graph(nodes, num_edges, graph_type):
    #Create a new Pydot graph object with the specified graph type
    graph = pydot.Dot(graph_type=graph_type, strict=True)

    #Add nodes to the graph
    for node in nodes:
        #Create a new node with the current node name and add it to the graph
        graph.add_node(pydot.Node(name=node))

    #Generating random edges
    edge_set = set()                                #Declaring Set to store generated edges, since it doesnot have duplicates
    while len(edge_set) < num_edges:
        #Randomly select a source and target node from the list of nodes
        node = random.choice(nodes)
        adj_node = random.choice(nodes)
        #checking node not equal to adj_node and also checking the edge is not in set
        if node != adj_node and (node, adj_node) not in edge_set and (adj_node, node) not in edge_set:
            edge_set.add((node, adj_node))    #Add the edge to the set
    
    #print(edge_set)
    #Add edges to the graph
    for edge in edge_set:
                                                               #Add an edge between the selected source and target nodes
        graph.add_edge(pydot.Edge(edge[0], edge[1]))

    return graph                                               #Return the generated random graph

def main():
    nodes = ["A", "B", "C", "D", "E", "F", "G","Z"]                #giving list of nodes
    num_edges = 8                                              #initializing num of edges
    graph_type = 'digraph'
 
    graph = random_graph(nodes, num_edges, graph_type)        #calling random__graph
    graph.write_png("random_graph.png")                       #writing graph to random_graph.png

if __name__ == "__main__":                                    #checks if the script is being executed directly by the Python or imported as a module.
    main()                                                    #Calling the main function
