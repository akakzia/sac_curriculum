
from collections import defaultdict
import random

from graph.semantic_graph import SemanticGraph
import networkit as nk

class Teacher():
    def __init__(self,args):
        self.oracle_graph = SemanticGraph.load_oracle(args.n_blocks)
        self.args = args
        self.agent_frontier = {} # store configuration through networkit node_id from agent_graph 

    def is_in_frontier(self,config,agent_graph:SemanticGraph):
        '''
        Compute the ensemble of nodes wich a part of the frontier : 
            --> nodes that exist in the oracle graph
            --> nodes that are not intermediate node in the path [coplanar -> any node] of the agent graph
            --> nodes that have unknown explorable childs 
        '''
        if not self.oracle_graph.hasNode(config):
            return False

        if agent_graph.getNodeId(config) in agent_graph.frontier:
            return True
    
        neighbours = self.oracle_graph.iterNeighbors(config)
        for neighbour in neighbours:
            # if not agent_graph.hasNode(neighbour):
            if not agent_graph.hasEdge(config, neighbour):
                return True
        return False

    def computeFrontier(self,agent_graph:SemanticGraph):
        self.agent_frontier = set()
        for node in agent_graph.configs:
            if self.is_in_frontier(node,agent_graph):
                self.agent_frontier.add( agent_graph.getNodeId(node))
        
    def sample_in_frontier(self,current_node,agent_graph,k):
        reachables = agent_graph.get_reachables_node_ids(current_node)
        reachable_frontier = [agent_graph.getConfig(node_id) 
                              for node_id in reachables 
                              if node_id in self.agent_frontier] 
        if reachable_frontier:
            return random.choices(reachable_frontier,k=k) # sample with replacement
        else: 
            return []

    def sample_from_frontier(self,node,agent_graph,k):
        to_explore = []
        for neighbour in self.oracle_graph.iterNeighbors(node):
            if not agent_graph.hasEdge(node,neighbour):
                to_explore.append(neighbour)
        if to_explore:
            return random.choices(to_explore,k=k) # sample with replacement
        else : 
            return []

    def sample_goal_uniform(self,nb_goal):
        return random.choices(self.oracle_graph.configs.inverse,k=nb_goal) # sample with replacement
