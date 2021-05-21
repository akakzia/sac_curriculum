from graph.SemanticOperation import SemanticOperation,config_permutations
import networkit as nk
import pickle
from bidict import bidict
import copy 

class SemanticGraph:
    def __init__(self,configs : bidict,graph :nk.graph,nb_blocks,GANGSTR=True):
        self.configs = configs
        self.nk_graph = graph
        self.nb_blocks = nb_blocks
        self.GANGSTR = GANGSTR
        self.semantic_operation = SemanticOperation(nb_blocks,True)
    
    def save(self,suffix=''):
        writer = nk.Format.NetworkitBinary
        nk.writeGraph(self.nk_graph,f"data/oracle_graph_block{self.nb_blocks}_{suffix}.nk", writer)
        with open(f"data/oracle_configs_block{self.nb_blocks}_{suffix}.config", 'wb') as f:
            pickle.dump(self.configs,f,protocol=pickle.HIGHEST_PROTOCOL)

    def load(nb_blocks,suffix=''):
        with open(f"data/oracle_configs_block{nb_blocks}_{suffix}.config",'rb') as f:
            configs = pickle.load(f)
        reader = nk.Format.NetworkitBinary
        nk_graph = nk.readGraph(f"data/oracle_graph_block{nb_blocks}_{suffix}.nk", reader)
        return SemanticGraph(configs,nk_graph,nb_blocks)

    def get_path_from_coplanar(self,goal):
        return self.get_path(self.semantic_operation.empty(),goal)

    def get_path(self,c1,c2):
        c1,c2 = tuple(c1),tuple(c2)
        try :
            n1 = self.configs[c1]
            n2 = self.configs[c2]
            dijkstra = nk.distance.Dijkstra(self.nk_graph, n1, True, False, n2)
            dijkstra.run()
            config_path =  [self.configs.inverse[node] for node in  dijkstra.getPath(n2)]
        except KeyError:
            config_path = [c1,c2]
        return config_path
    
    def get_frontiere_from_configs(self,configs):
        nexts = []
        isolated = []
        for c in configs : 
            if self.nk_graph.isIsolated(self.configs[c]):
                isolated.append(self.configs[c])
            nexts += [neighbour for neighbour in self.nk_graph.iterNeighbors(c)
                                if neighbour not in configs] 
        return nexts

    def getNode(self,config):
        return self.configs[config]

    def getConfig(self,nodeId):
        return self.configs.inverse[nodeId]
    
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
        






