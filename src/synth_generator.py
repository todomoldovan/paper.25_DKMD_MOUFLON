import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import pickle
import time
import gc

# Global paths
obj_path="../data/obj"

def gen_my_networks(ex_id):
	G = nx.Graph()
	elist = []
	
	# Network 1
	if ex_id == 1:
		G.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11])
		elist = [
			# Friend group clique
			(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
			(1,2),(1,3),(1,4),(1,5),(1,6),
			(2,3),(2,4),(2,5),(2,6),
			(3,4),(3,5),(3,6),
			(4,5),(4,6),
			(5,6),
			# Indian community
			(7,8),(7,9),(7,10),(7,11),
			(8,9),(8,10),(8,11),
			(10,11),
			# Extra edges
			(0,7),(0,8),
			(1,8),
			(6,7),(6,8)
		]
	elif ex_id == 2:
		G.add_nodes_from([0,5,6,8,9],color="red")
		G.add_nodes_from([1,2,3,4,7],color="blue")
		elist = [
			# Anand group clique
			(0,1),(0,2),(0,3),
			(1,2),(1,3),
			(2,3),
			# Thea group
			(4,5),(4,6),
			(5,6),
			# Ella group
			(7,8),(7,9),
			(8,9),
			# Extra edges
			(0,7),
			(7,4),
			(4,3)
		]
	elif ex_id == 3:
		G.add_nodes_from([0,5,6,8,9],color="red")
		G.add_nodes_from([1,2,3,4,7],color="blue")
		elist = [
			# Anand group clique
			(0,1),(0,2),(0,3),
			(1,2),(1,3),
			(2,3),
			# Thea group
			(4,5),(4,6),
			(5,6),
			# Ella group
			(7,8),(7,9),
			(8,9),
			# Extra edges
			(0,7),(0,4),
			(7,4),
			(4,3),(3,7)
		]
	if ex_id == 4:
		G.add_nodes_from([0,1,7,8,9,10,11],color="red")
		G.add_nodes_from([2,3,4,5,6],color="blue")
		elist = [
			# Friend group clique
			(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
			(1,2),(1,3),(1,4),(1,5),(1,6),
			(2,3),(2,4),(2,5),(2,6),
			(3,4),(3,5),(3,6),
			(4,5),(4,6),
			(5,6),
			# Indian community
			(7,8),(7,9),(7,10),(7,11),
			(8,9),(8,10),(8,11),
			(10,11),
			# Extra edges
			(0,7),(0,8),
			(1,8),
			(6,7),(6,8)
		]
	elif ex_id == 5:
		G.add_nodes_from([0,5,6,7,8],color="red")
		G.add_nodes_from([1,2,3,4],color="blue")
		elist = [
			# Anand group clique
			(0,1),(0,2),
			(1,2),
			# Thea group
			(3,4),(3,5),
   			(4,5),
			# Ella group
			(7,8),(7,6),
			(8,6),
			# Extra edges
			(0,3),(0,6),
			(6,3)
		]
		
	# Add simple weights
	for e in elist:
		G.add_edge(*e,weight=1.0)

	# Save object
	with open(f"{obj_path}/network{ex_id}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)

	return G

def simple_synth(ex_id):  
	G = nx.Graph()
	elist=[]

	# Example 1: two fully connected 6-node cliques, 1R 1B. Two edges between them
	if ex_id==1:
		# Add nodes. Even ids are red, odd ids are blue
		G.add_nodes_from([0,2,4,6,8,10],color="red")
		G.add_nodes_from([1,3,5,7,9,11],color="blue")
		# Edgelist
		elist = [
			# red clique
			(0,2),(0,4),(0,6),(0,8),(0,10),
			(2,4),(2,6),(2,8),(2,10),
			(4,6),(4,8),(4,10),
			(6,8),(6,10),
			(8,10),
			# blue clique
			(1,3),(1,5),(1,7),(1,9),(1,11),
			(3,5),(3,7),(3,9),(3,11),
			(5,7),(5,9),(5,11),
			(7,9),(7,11),
			(9,11),
			# extra edges
			(1,8),(3,4)
		]

	# Example 2: two fully connected 6-node cliques, nodes are mixed (5/1). 
	elif ex_id==2:
		# Add nodes
		G.add_nodes_from([0,1,2,3,4,11],color="red")
		G.add_nodes_from([5,6,7,8,9,10],color="blue")
		# Edgelist
		elist = [
			# red heavy clique
			(0,1),(0,2),(0,3),(0,4),(0,5),
			(1,2),(1,3),(1,4),(1,5),
			(2,3),(2,4),(2,5),
			(3,4),(3,5),
			(4,5),
			# blue heavy clique
			(6,7),(6,8),(6,9),(6,10),(6,11),
			(7,8),(7,9),(7,10),(7,11),
			(8,9),(8,10),(8,11),
			(9,10),(9,11),
			(10,11),
			# extra edges
			(1,8),(3,9)
		]

	# Example 3: three fully connected 4-node cliques, 1R 1B 1 mixed (0.5/0.5).
	elif ex_id==3:    
		# Add nodes
		G.add_nodes_from([0,1,2,3,4,5],color="red")
		G.add_nodes_from([6,7,8,9,10,11],color="blue")
		# Edgelist
		elist = [
			# clique 1 (red)
			(0,1),(0,2),(0,3),
			(1,2),(1,3),
			(2,3),
			# clique 2 (mixed 2/2)
			(4,5),(4,6),(4,7),
			(5,6),(5,7),
			(6,7),
			# clique 3 (blue)
			(8,9),(8,10),(8,11),
			(9,10),(9,11),
			(10,11),
			# extra edges
			(1,5),(6,10),(0,11)
		]

	# Add simple weights
	for e in elist:
		G.add_edge(*e,weight=1.0)

	# Save object
	with open(f"{obj_path}/example_{ex_id}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)

	return G

# Use revamped for weight issue while running experiments
#K = 12 # number of clusters
#l = 5 # size of clusters
def full_clique_colored_evolving(K,l,p,colors=["blue","red"],prob_colors=[0.5,0.5]):
	'''
	K: number of clusters
	l: size of clusters
	p: ratio of changing edges
	'''

	# Create an empty graph
	G = nx.Graph()

	# Create K copies of complete graph on l nodes and add them to G
	complete = nx.complete_graph(l)
	clusters = []
	for k in range(K):
		cluster = nx.relabel_nodes(complete, {i: l * k + i for i in range(l)})#relabel node id
		
		# For each cluster, pick a color for all its nodes
		cluster_cols={}
		# By default, pick last colour in list (in case the probabilities dont sum to 1)
		picked_col=colors[len(colors)-1]
		# Get random number here drawn from uniform
		c_prob=random.uniform(0.0,1.0)
		cp_sum=0.0
		# Pick colour from cumulative sum of probabilities
		for c_ind,col in enumerate(colors):
			cp_sum+=prob_colors[c_ind]
			# Pick colour if probability is less than the cumulative sum of colour prob
			if c_prob<=cp_sum:
				picked_col=colors[c_ind]
				break
		# Set all nodes in cluster to this color. Index: l*k+i for i in range(l)
		nx.set_node_attributes(cluster,picked_col,"color")

		G = nx.compose(G, cluster)# compose complete graph
		clusters.append(cluster)# add complete graph

	N = len(G.nodes)
	M = len(G.edges)

	# Change the edges
	k = 0
	for _ in range(int(p* M)):
		(I, J) = random.choice(list(clusters[k].edges)) #choose edge randomly from cluster k
		clusters[k].remove_edge(I, J) #remove edge from clusters
		G.remove_edge(I, J) #remove edge from G
		A = random.choice([I, J]) #random select one side of edge
		while(True):
			B = (l * (k + 1)  + random.randrange(l * (K - 1))) % N #random select another side from other clusters
			if A == B:
				continue
			if G.has_edge(A, B):
				continue
			G.add_edge(A, B)
			break
		k += 1
		k %= K #keep k<K
	
	wei=[]
	g=nx.Graph()
	# Add here for individual node probabilities
	edge=list(G.edges())
	print(edge)

	for i in range(len(edge)):
		wei.append((edge[i][0],edge[i][1],1.0))
	g.add_weighted_edges_from(wei)

	# Save object. Name format: {nsize=K*l}_p{rewire_prob=P}_K{k_colours}_c{prob of last/minority colour}
	rew_prob_to_name=("".join(str(p).split(".")))
	color_prob_to_name=("".join(str(prob_colors[len(colors)-1]).split(".")))
	with open(f"{obj_path}/color-full_{(K*l)}_r{rew_prob_to_name}_K{len(colors)}_c{color_prob_to_name}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)

	return g


def prob_clique_colored_evolving(K,l,p,colors=["blue","red"],prob_colors=[0.5,0.5]):
	'''
	K: number of clusters
	l: size of clusters
	p: ratio of changing edges
	'''

	# Create an empty graph
	G = nx.Graph()

	# Create K copies of complete graph on l nodes and add them to G
	complete = nx.complete_graph(l)
	clusters = []
	for k in range(K):
		cluster = nx.relabel_nodes(complete, {i: l * k + i for i in range(l)})#relabel node id
		
		# For each cluster, pick a color for all its nodes
		cluster_cols={}

		# For each node in the cluster: pick individual color
		for n in cluster.nodes():
			# By default, pick last colour in list (in case the probabilities dont sum to 1)
			picked_col=colors[len(colors)-1]
			# Get random number here drawn from uniform
			c_prob=random.uniform(0.0,1.0)
			cp_sum=0.0
			# Pick colour from cumulative sum of probabilities
			for c_ind,col in enumerate(colors):
				cp_sum+=prob_colors[c_ind]
				# Pick colour if probability is less than the cumulative sum of colour prob
				if c_prob<=cp_sum:
					picked_col=colors[c_ind]
					break
			# Add picked color to individual node
			cluster_cols[n]={"color":picked_col}

		# Set all nodes in cluster to this color. Index: l*k+i for i in range(l)
		nx.set_node_attributes(cluster,cluster_cols)

		G = nx.compose(G, cluster)# compose complete graph
		clusters.append(cluster)# add complete graph

	N = len(G.nodes)
	M = len(G.edges)

	# Change the edges
	k = 0
	for _ in range(int(p* M)):
		(I, J) = random.choice(list(clusters[k].edges)) #choose edge randomly from cluster k
		clusters[k].remove_edge(I, J) #remove edge from clusters
		G.remove_edge(I, J) #remove edge from G
		A = random.choice([I, J]) #random select one side of edge
		while(True):
			B = (l * (k + 1)  + random.randrange(l * (K - 1))) % N #random select another side from other clusters
			if A == B:
				continue
			if G.has_edge(A, B):
				continue
			G.add_edge(A, B)
			break
		k += 1
		k %= K #keep k<K
  
	# Set weight of all edges of G to 1.0
	nx.set_edge_attributes(G, 1.0, "weight")
	
	wei=[]
	# g=nx.Graph()
	# # Add here for individual node probabilities
	# edge=list(G.edges())
	# for i in range(len(edge)):
	# 	wei.append((edge[i][0],edge[i][1],1.0))
	# g.add_weighted_edges_from(wei)

	# Save object. Name format: {nsize=K*l}_p{rewire_prob=P}_K{k_colours}_c{prob of last/minority colour}
	rew_prob_to_name=("".join(str(p).split(".")))
	color_prob_to_name=("".join(str(prob_colors[len(colors)-1]).split(".")))
	with open(f"{obj_path}/color-node_{(K*l)}_r{rew_prob_to_name}_K{len(colors)}_c{color_prob_to_name}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)

	return G

def full_clique_colored_evolving_revamped(K,l,p,colors=["blue","red"],prob_colors=[0.5,0.5]):
	'''
	K: number of clusters
	l: size of clusters
	p: ratio of changing edges
	'''

	# Create an empty graph
	G = nx.Graph()

	# Create K copies of complete graph on l nodes and add them to G
	complete = nx.complete_graph(l)
	clusters = []
	for k in range(K):
		cluster = complete.copy()
		cluster = nx.relabel_nodes(cluster, {i: l * k + i for i in range(l)})	#relabel node id
		
		# For each cluster, pick a color for all its nodes
		cluster_cols={}
		# Randomly pick color for clusters
		picked_col = random.choices(colors, weights=prob_colors, k=1)[0]
		# cluster_cols = {node: picked_col for node in cluster.nodes()}
		nx.set_node_attributes(cluster,picked_col,"color")

		G = nx.compose(G, cluster)# compose complete graph
		clusters.append(cluster)# add complete graph

	N = G.number_of_nodes()	# len(G.nodes)
	M = G.number_of_edges()	# len(G.edges)
	num_rewire = int(p * M)
 
	# Get all cluster edges as a list for efficient sampling
	cluster_edges = [list(cluster.edges) for cluster in clusters]

	# Change the edges
	for _ in range(num_rewire):
		k = random.randrange(K)		# Pick a random cluster
		if not cluster_edges[k]:	# Skip if no cluster edges remain
			continue

		# Randomly pick an edge and remove it
		(I, J) = random.choice(cluster_edges[k])
		G.remove_edge(I, J)
		cluster_edges[k].remove((I, J))
  
		# Select a new edge from another cluster
		A = random.choice([I, J])
		while(True):
			B = random.randint(0, N-1)
			if A != B and not G.has_edge(A, B):
				G.add_edge(A, B)
				break

	# Set weight of all edges of G to 1.0
	nx.set_edge_attributes(G, 1.0, "weight")
	
	# g=nx.Graph()
	# g.add_weighted_edges_from((u, v, 1.0) for u, v in G.edges())
 
	g = G.copy()	# Create a new graph with the same edges and weights

	# Save object. Name format: {nsize=K*l}_p{rewire_prob=P}_K{k_colours}_c{prob of last/minority colour}
	rew_prob_to_name=("".join(str(p).split(".")))
	color_prob_to_name=("".join(str(prob_colors[len(colors)-1]).split(".")))
	with open(f"{obj_path}/color-full_{(K*l)}_r{rew_prob_to_name}_K{len(colors)}_c{color_prob_to_name}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)

	return g

def generate_erdos_renyi_graph(n_nodes, density_prob, red_prob=0.5, seed=42):
	"""
	Generate an Erdos-Renyi graph with weighted edges and colored nodes.

	Parameters:
		n_nodes (int): Number of nodes in the graph.
		density (float): Probability of edge creation (between 0 and 1).
		red_prob (float): Probability of assigning a node the color 'red'.
							(1 - red_prob) will be the probability for 'blue'.
		seed (int or None): Random seed for reproducibility.

	Returns:
		G (networkx.Graph): A graph with node colors and edge weights.
	"""
	# Create Erdos-Renyi graph
	# G = nx.erdos_renyi_graph(n_nodes, density_prob, seed=seed)
	G = nx.fast_gnp_random_graph(n_nodes, density_prob, seed=seed)

	# Assign weight of 1 to each edge
	for u, v in G.edges():
		G[u][v]['weight'] = 1.0
  
	# Assign colors to nodes based on red_prob
	for node in G.nodes():
		G.nodes[node]['color'] = 'red' if random.random() < red_prob else 'blue'
  
	# Save object. Name format: {nsize=K*l}_p{rewire_prob=P}_K{k_colours}_c{prob of last/minority colour}
	rew_prob_to_name=("".join(str(density_prob).split(".")))
	color_prob_to_name=("".join(str(red_prob).split(".")))
	with open(f"{obj_path}/{n_nodes}/ER_{(n_nodes)}_r{rew_prob_to_name}_K2_c{color_prob_to_name}.nx","wb") as tw_out:
		pickle.dump(G,tw_out)
  
	return G



'''
# Define custom positions for the vertices
positions = {
	l * k + i: (
		math.sin(k / K * math.pi * 2) + math.sin(i / l * math.pi * 2) / K * 2,
		math.cos(k / K * math.pi * 2) + math.cos(i / l * math.pi * 2) / K * 2,
	) for k in range(K) for i in range(l)
}

# Draw the graph with custom positions
plt.figure(figsize=(8, 6))
nx.draw(
	G, pos=positions, with_labels=True,
	node_color='lightgreen', node_size=800,
	edge_color='blue', font_weight='bold'
)

plt.title("Custom Graph with 10 Vertices")
plt.show()
'''




# # Generate synth examples
# for ex_id in [1,2,3]:
# 	G = simple_synth(ex_id)
# 	nx.draw(G, with_labels=True)
# 	plt.show()

# """
# Generate evolving examples, full clique coloured. Adjust params accordingly.
# 	Example now generates test networks for 5 cliques of 5 nodes, 
# 	prob. of rewiring=0.1, 2 colours with probs. 
# 	blue=1-prob_sensitive, red=prob_sensitive
# """
# for prob_sensitive in [0.1,0.2,0.3,0.4,0.5]:
# 	full_clique_colored_evolving(
# 		5, 												# K cliques
# 		5, 												# clique size
# 		0.2, 											# edge rewiring probability
# 		colors=["blue","red"], 							# color list
# 		prob_colors=[1-prob_sensitive,prob_sensitive] 	# prob of colours
# 	)


# """
# Generate evolving examples, individual nodes coloured. Adjust params accordingly.
# 	Example now generates test networks for 5 cliques of 5 nodes, 
# 	prob. of rewiring=0.1, 2 colours with probs. 
# 	blue=1-prob_sensitive, red=prob_sensitive
# """
# for prob_sensitive in [0.1,0.2,0.3,0.4,0.5]:
# 	prob_clique_colored_evolving(
# 		5, 												# K cliques
# 		5, 												# clique size
# 		0.2, 											# edge rewiring probability
# 		colors=["blue","red"], 							# color list
# 		prob_colors=[1-prob_sensitive,prob_sensitive] 	# prob of colours
# 	)


# # G = gen_my_networks(5)
# start = time.time()
# # G = full_clique_colored_evolving(10, 100, 0.1, colors=["blue","red"], prob_colors=[0.5,0.5])
# G = full_clique_colored_evolving_revamped(10, 500, 0.2, colors=["blue", "red"], prob_colors=[0.5, 0.5])
# end = time.time()
# print(f"Time taken: {end-start} s")
# nx.draw(G, with_labels=True)
# plt.show()

start = time.time()
# G = generate_erdos_renyi_graph(1000, 0.001, red_prob=0.5, seed=42)
# G = full_clique_colored_evolving_revamped(4, 5, 0.1, colors=["blue", "red"], prob_colors=[0.5, 0.5])
G = prob_clique_colored_evolving(4, 5, 0.1, colors=["blue", "red"], prob_colors=[0.5, 0.5])
end = time.time()
print(f"Time taken for 0.1: {end-start} s")
print(set(nx.get_node_attributes(G, 'color').values()))
node_colors = [G.nodes[node]["color"] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500)
plt.show()
del G
gc.collect()
