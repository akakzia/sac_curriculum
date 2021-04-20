
from itertools import permutations
from utils import get_graph_structure

class SemanticOperation():
    '''
    Class used to transform configuration with above and close operations. 
    '''
    def __init__(self,nb_blocks,GANGSTR):
        self.nb_blocks = nb_blocks
        edges, incoming_edges, predicate_ids = get_graph_structure(nb_blocks)
        self.edge_to_close_ids = { edge:predicate_ids[edge_id][0] for edge_id,edge in enumerate(edges)  }
        self.edge_to_above_ids = { edge:predicate_ids[edge_id][1] for edge_id,edge in enumerate(edges)  }

        # define how truth values are replaced in semantic configurations : 
        if GANGSTR:
            self.semantic = {True: 1., False:-1.}
        else : 
            self.semantic = {True: 1., False:0.}

    def close(self,config,a,b,pred_val):
        '''Return a copy of config with a close from b with value True/False'''
        above_id = self.edge_to_close_ids[(a,b)]
        new_config = config[:above_id] + (self.semantic[pred_val],) + config[above_id+1:]
        return new_config

    def above(self,config,a,b,pred_val):
        '''Return a copy of config with a above b with value True/False'''
        above_id = self.edge_to_above_ids[(a,b)]
        new_config = config[:above_id] + (self.semantic[pred_val],) + config[above_id+1:]
        return new_config

    def close_and_above(self,config,a,b,pred_val):
        '''Return a copy of config with a above b and a close from b with value True/False'''
        config = self.close(config,a,b,pred_val)
        return self.above(config,a,b,pred_val)
    
    def empty(self):
        ''' Return the empty configuration where everything is far appart'''
        return (self.semantic[False],) * ((3*self.nb_blocks * (self.nb_blocks-1))//2)
    
    def to_GANGSTR(config):
        tuple(1 if c > 0 else -1 
                        for c in config)

def all_stack_trajectories(stack_size,GANGSTR= True):
    '''Return a dictionnary of cube-stack associated with semantic-trajectories :
    Keys are all possible stack permutation. (ex : (0,1,2) , (1,2,0) ... )
    Values contains all intermediate configurations from [0,0..] to the config describing the key.'''
    # Generate all possible stack list
    all_cubes = range(stack_size)
    all_stack = list(permutations(all_cubes))

    sem_op = SemanticOperation(stack_size,GANGSTR)
    config_to_path = {}
    for stack in all_stack:
        cur_config = sem_op.empty() # start with the empy config [0,0, ... ]
        config_path = [] 
        # construct intermediate stack config by adding blocks one by one : 
        for top,bottom in zip(stack,stack[1:]) : 
            cur_config = sem_op.close_and_above(cur_config,bottom,top,1)
            config_path.append(cur_config)
        config_to_path[stack] = config_path

    return config_to_path

