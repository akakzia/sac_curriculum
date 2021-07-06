from collections import defaultdict
import math
import numpy as np
from bidict import bidict
import networkit as nk
from graph.SemanticOperation import config_permutations
from graph.semantic_graph import SemanticGraph

class UnorderedSemanticGraph(SemanticGraph):
    '''
    all edges identical to one permutation share the same success rate
    '''
    def __init__(self, configs: bidict, graph: nk.graph, nb_blocks, GANGSTR=True,edges_infos=None,args=None):
        super().__init__(configs, graph, nb_blocks, GANGSTR=GANGSTR, edges_infos=edges_infos, args=args)
        self.ordered_edge_to_unordered_edge = dict()
        self.unordered_edge_to_ordered_edge = defaultdict(set)

    def create_node(self,config):
        if config not in self.configs:
            for c in set(config_permutations(config,self.semantic_operation)):
                super().create_node(c)
    
    def create_edge_stats(self,edge,start_sr):
        c1,c2 = edge

        if self.args.one_object_edge and not self.semantic_operation.one_object_edge(edge):
            return

        if not self.hasEdge(c1,c2):
            unordered_id = len(self.unordered_edge_to_ordered_edge)
            self.edges_infos[unordered_id] = {'SR':start_sr,'Count':1}

            clamped_sr = max(np.finfo(float).eps, min(start_sr, 1-np.finfo(float).eps))
            additive_sr = -math.log(clamped_sr)
            
            for c_perm_1,c_perm_2 in zip(config_permutations(c1,self.semantic_operation),
                                        config_permutations(c2,self.semantic_operation)):
                n1,n2 = self.configs[c_perm_1],self.configs[c_perm_2]
                if not self.nk_graph.hasEdge(n1,n2):
                    self.nk_graph.addEdge(n1,n2)
                    self.nk_graph.setWeight(n1,n2,additive_sr)

                    self.ordered_edge_to_unordered_edge[(n1,n2)] = unordered_id
                    self.unordered_edge_to_ordered_edge[unordered_id].add((n1,n2))
        else : 
            raise Exception(f'Already existing edge {c1}->{c2}')

    def edge_config_to_edge_id(self,edge_config):
        c1,c2 = edge_config
        n1,n2 = (self.configs[c1],self.configs[c2])
        unordered_edge_id = self.ordered_edge_to_unordered_edge[(n1,n2)]
        return unordered_edge_id

    def update_graph_edge_weight(self,edge):

        if self.args.one_object_edge and not self.semantic_operation.one_object_edge(edge):
            return

        new_mean_sr = self.edges_infos[edge]['SR']
        clamped_sr = max(np.finfo(float).eps, min(new_mean_sr, 1-np.finfo(float).eps))

        for n1,n2 in self.unordered_edge_to_ordered_edge[edge]:
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))

    def log(self,logger):
        logger.record_tabular('agent_nodes_ordered',self.nk_graph.numberOfNodes())
        logger.record_tabular('agent_edges_ordered',self.nk_graph.numberOfEdges())
        logger.record_tabular('agent_edges_unordered',len(self.edges_infos))
