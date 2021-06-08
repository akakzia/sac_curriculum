
import random

from graph.semantic_graph import SemanticGraph

class Teacher():
    def __init__(self,args):
        self.oracle_graph = SemanticGraph.load_oracle(args.n_blocks)
        self.args = args
        self.agent_frontier = [self.oracle_graph.empty()]

    def is_in_frontier(self,config,agent_graph:SemanticGraph):
        '''
        Compute the ensemble of nodes wich a part of the frontier : 
            --> nodes that have no childs
            --> nodes that have unknown explorable childs 
        '''
        if not self.oracle_graph.hasNode(config):
            return False

        if self.oracle_graph.getNodeId(config) in self.oracle_graph.frontier:
            return True
    
        neighbours = self.oracle_graph.iterNeighbors(config)
        for neighbour in neighbours:
            # if not agent_graph.hasNode(neighbour):
            if not agent_graph.hasEdge(config, neighbour):
                return True
        return False

    def computeFrontier(self,agent_graph:SemanticGraph):
        self.agent_frontier = []
        for node in agent_graph.configs:
            if self.is_in_frontier(node,agent_graph):
                self.agent_frontier.append(node)
        
    def sample_in_frontier(self,k):
        return random.choices(self.agent_frontier,k=k) # sample with replacement

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
