import os.path
import copy 
from collections import defaultdict
import math
import numpy as np
import pickle
from bidict import bidict
import networkit as nk
from graph.SemanticOperation import SemanticOperation,config_permutations

class SemanticGraph:

    ORACLE_PATH = 'data/'
    ORACLE_NAME = 'oracle_block'

    def __init__(self,configs : bidict,graph :nk.graph,nb_blocks,GANGSTR=True,edges_infos=None,args=None):
        self.configs = configs
        if edges_infos == None:
            self.edges_infos = defaultdict(dict)
        else : 
            self.edges_infos = edges_infos
        self.nk_graph = graph
        self.nb_blocks = nb_blocks
        self.GANGSTR = GANGSTR
        self.args = args
        self.semantic_operation = SemanticOperation(nb_blocks,True)

        self.dijkstra_from_coplanar = nk.distance.Dijkstra(self.nk_graph,self.configs[self.empty()], True, False)
        self.dijkstra_from_coplanar.run()
        self.frontier = set(self.get_frontier_nodes())
    
    def save(self,path,name):
        writer = nk.Format.NetworkitBinary
        graph_filename = f"{path}graph_{name}.nk"
        if os.path.isfile(graph_filename):
                os.remove(graph_filename)
        nk.writeGraph(self.nk_graph,graph_filename, writer)
        with open(f"{path}configs_{name}.config", 'wb') as f:
            pickle.dump(self.configs,f,protocol=pickle.HIGHEST_PROTOCOL)
        if len(self.edges_infos)>0:
            with open(f"{path}edges_{name}.infos", 'wb') as f:
                pickle.dump(self.edges_infos,f,protocol=pickle.HIGHEST_PROTOCOL)

    def load(path:str,name:str,nb_blocks:int,args=None):
        with open(f"{path}configs_{name}.config", 'rb') as f:
            configs = pickle.load(f)
        if os.path.isfile(f"{path}edges_{name}.infos"):
            with open(f"{path}edges_{name}.infos", 'rb') as f:
                edges_infos = pickle.load(f)
        else : 
            edges_infos = None
        reader = nk.Format.NetworkitBinary
        nk_graph = nk.readGraph(f"{path}graph_{name}.nk", reader)
        return SemanticGraph(configs,nk_graph,nb_blocks,edges_infos=edges_infos,args=args)

    def load_oracle(nb_blocks:int):
        return SemanticGraph.load(SemanticGraph.ORACLE_PATH,
                                f'{SemanticGraph.ORACLE_NAME}{nb_blocks}',nb_blocks)

    def update_shortest_tree(self):
        self.dijkstra_from_coplanar.run()

    def distance_from_coplanar(self,target):
        return self.dijkstra_from_coplanar.distance(self.empty(),target)

    def get_path_from_coplanar(self,goal):
        return self.get_path(self.semantic_operation.empty(),goal)

    def get_path(self,c1,c2):
        c1,c2 = tuple(c1),tuple(c2)
        distance = None
        try :
            n1 = self.configs[c1]
            n2 = self.configs[c2]
            dijkstra = nk.distance.Dijkstra(self.nk_graph, n1, True, False, n2)
            dijkstra.run()
            config_path =  [self.configs.inverse[node] for node in  dijkstra.getPath(n2)]
            if config_path:
                distance = np.exp(-dijkstra.distance(n2))
        except KeyError:
            config_path = []
        return config_path,distance
        
    def sample_path(self,c1,c2,k):
        raise NotImplementedError()
            
    def get_isolated_nodes(self):
        isolated = []
        for c in self.nk_graph.iterNodes():
            if self.nk_graph.isIsolated(c):
                isolated.append(c)
        return isolated
    

    def get_reachables_node_ids(self,source):
        reachables = []
        if source in self.configs:
            source_id = self.configs[source]
            bfs = nk.distance.BFS(self.nk_graph, source_id, True, True)
            bfs.run()
            reachables = bfs.getNodesSortedByDistance()
        return reachables



    def get_frontier_nodes(self):
        self.dijkstra_from_coplanar.run()
        intermediate_nodes = set()
        for node in self.nk_graph.iterNodes():
            predecessors = self.dijkstra_from_coplanar.getPredecessors(node)
            if predecessors : 
                intermediate_nodes.update(predecessors)
        isolated = []
        for node in self.nk_graph.iterNodes():
            if node not in intermediate_nodes:
                isolated.append(node)
        return isolated

    def add_config(self,config):
        if config not in self.configs:
            self.configs[config] = self.nk_graph.addNode()


    def create_edge(self,edge,start_sr):
        c1,c2 = edge
        n1,n2 = self.configs[c1],self.configs[c2]
        if not self.nk_graph.hasEdge(n1,n2):
            self.nk_graph.addEdge(n1,n2)
            self.edges_infos[(n1,n2)] = {'SR':start_sr,'Count':1}
            clamped_sr = max(np.finfo(float).eps, min(start_sr, 1-np.finfo(float).eps))
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))
        else : 
            raise Exception(f'Already existing edge {n1}->{n2}')

    def update_edge(self,edge,success):
        c1,c2 = edge
        n1,n2 = self.configs[c1],self.configs[c2]
        success = int(success)
        
        if not self.nk_graph.hasEdge(n1,n2):
            raise Exception(f"unknown edge {n1}->{n2}")
        else:
            # update SR  :
            self.edges_infos[(n1,n2)]['Count']+=1
            count = self.edges_infos[(n1,n2)]['Count']
            last_mean_sr = self.edges_infos[(n1,n2)]['SR']
            if self.args.edge_sr == 'moving_average':
                new_mean_sr = last_mean_sr + (1/count)*(success-last_mean_sr)
            elif self.args.edge_sr == 'exp_moving_average':
                new_mean_sr = self.args.edge_lr* last_mean_sr + (1-self.args.edge_lr)*(success)
            else : 
                raise Exception(f"Unknown self.args.edge_sr value : {self.args.edge_sr}")
            self.edges_infos[(n1,n2)]['SR'] = new_mean_sr

            clamped_sr = max(np.finfo(float).eps, min(new_mean_sr, 1-np.finfo(float).eps))
            # weight is set to -log(SR) because Djikstra is used for shortest-path algorithm
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))

    def hasNode(self,config):
        if config in self.configs:
            return self.nk_graph.hasNode(self.configs[config])
        return False

    def hasEdge(self,config_start,config_end):
        if config_start in self.configs and config_end in self.configs:
            return self.nk_graph.hasEdge(self.configs[config_start],
                                         self.configs[config_end])
        return False

    def iterNeighbors(self,config):
        '''iter over neighbors of a node, take in a semantic config
            return a generator over semantic configs.'''
        if config in self.configs:
            return (self.configs.inverse[node_id] for node_id in self.nk_graph.iterNeighbors(self.configs[config]))
        else : 
            return []

    def getNodeId(self,config):
        return self.configs.get(config,None)

    def getConfig(self,nodeId):
        return self.configs.inverse[nodeId]

    def getWeight(self,c1,c2):
        return self.nk_graph.weight(self.getNodeId(c1),self.getNodeId(c2))
    
    def empty(self):
        return self.semantic_operation.empty()

def augment_with_all_permutation(nk_graph,configs,nb_blocks,GANGSTR=True):
    '''
    Takes a nk_graph as entry, configs wich translate configuration into node id. 
    configs is supposeed to only contains unique ordered configuration
    Return a new nk_graph and a new config dict with all ordered configurations.
    '''
    new_configs = copy.deepcopy(configs)
    new_nk_graph = copy.deepcopy(nk_graph)
    config_to_perms = dict()
    semantic_operator = SemanticOperation(nb_blocks,GANGSTR)

    # creates new nodes
    for config in configs:
        config_to_perms[config] = config_permutations(config,semantic_operator,nb_blocks)
        for config_perm in config_to_perms[config]:
            if config_perm not in new_configs:
                new_config_perm_id = new_nk_graph.addNode()
                new_configs[config_perm] = new_config_perm_id
                
    # creates new edges
    for config,config_perms in config_to_perms.items():
        for perm_id,config_perm in enumerate(config_perms):
            new_perm_id = new_configs[config_perm]
            for neighbour_id in nk_graph.iterNeighbors(configs[config]):
                perm_corresponding_neighbour = config_to_perms[new_configs.inverse[neighbour_id]][perm_id]
                perm_corresponding_id = new_configs[perm_corresponding_neighbour]
                if not new_nk_graph.hasEdge(new_perm_id,perm_corresponding_id):
                    new_nk_graph.addEdge(new_perm_id,perm_corresponding_id)

    return new_nk_graph,new_configs