

from collections import defaultdict
from graph.SemanticOperation import SemanticOperation, all_stack_trajectories, config_to_unique_str
from graph.semantic_graph import SemanticGraph
from graph.agent_network import AgentNetwork
from utils import get_eval_goals
import numpy as np
import math 
import unittest

def start_test(name):
    print()
    print('_'*10,name,'_'*10)
    print('_'*20 + '_'*(len(name)+2))

def check_generate_goals():
    ''' Check frontier goal sampling'''
    nb_blocks = 3
    GANGSTR = True
    semantic_operator = SemanticOperation(nb_blocks,GANGSTR)
    semantic_graph = SemanticGraph.load(nb_blocks)
    agentNetwork = AgentNetwork(semantic_graph,None)
    agentNetwork.add_goal(semantic_operator.empty())
    print([agentNetwork.semantic_graph.configs[c] for c in  agentNetwork.frontier])
    stack2 = semantic_operator.close_and_above(semantic_operator.empty(),0,1,True)
    agentNetwork.add_goal(stack2)
    print(agentNetwork.semantic_graph.getNode(stack2))
    print([agentNetwork.semantic_graph.configs[c] for c in  agentNetwork.frontier])

def check_stack_number(block_sizes):
    for nb_blocks in block_sizes:
        start_test(f"check_stack_number {nb_blocks}")
        # load graph 
        semantic_graph = SemanticGraph.load(nb_blocks)
        semantic_operator = SemanticOperation(nb_blocks,True)
        stacks = all_stack_trajectories(nb_blocks)
        all_good = True 
        for s,config_path in stacks.items():
            result,path_length = check_goal(semantic_graph,semantic_operator,config_path[-1],len(s)-1,s,True)
            if result != 'success': 
                all_good = False
        if all_good:
            print(f'Succeed all stacks of {nb_blocks}')

def test_eval_goals(block_sizes):
    for nb_blocks in block_sizes:
        start_test(f"test_eval_goals {nb_blocks}")
        # load graph 
        semantic_graph = SemanticGraph.load(nb_blocks)
        semantic_operator = SemanticOperation(nb_blocks,True)
        if nb_blocks == 3:
            instructions_to_length = {'close_1':1, 'close_2':2, 'close_3':3, 'stack_2':1, 'pyramid_3':2, 'stack_3':2}
        elif nb_blocks == 5:
            instructions_to_length = {'close_1':1, 'close_2':2, 'close_3':3, 'stack_2':1, 'stack_3':2, '2stacks_2_2':2, '2stacks_2_3':3, 'pyramid_3':2,
                            'mixed_2_3':None, 'trapeze_2_3':4, 'stack_4':3, 'stack_5':4}
        all_good = True 
        for instruction,true_length in instructions_to_length.items():
            error_count = defaultdict(int)
            goals = np.unique(get_eval_goals(instruction,nb_blocks,math.factorial(nb_blocks)*10),axis=0)
            for goal in goals:
                goal = tuple(goal)
                # if instruction == 'trapeze_2_3':
                #     print(config_to_unique_str(goal,semantic_operator))
                result,path_length = check_goal(semantic_graph,semantic_operator,goal,true_length,instruction)
                error_count[result]+=1
            if result=='missing_goal' : 
                print("missing_goal for ",instruction , f"({error_count['missing_goal']} failed /{len(goals)})")
                all_good = False
            elif result=='no_path' : 
                print("no_path ",instruction, f"({error_count['no_path']} failed /{len(goals)})")
                all_good = False
            elif result=='wrong_path' :
                print(f"wrong_path_size, should be {true_length}  ",instruction, f"({error_count['wrong_path_size']} failed /{len(goals)})")
        if all_good:
            print(f'Succeed tests {nb_blocks}')


def check_goal(semantic_graph,semantic_operator,goal,true_length,node_name,verbose=False):
    result = 'success'
    path_length = None
    if goal not in semantic_graph.configs:
        result = 'missing_goal'
    else : 
        path = semantic_graph.get_path(semantic_operator.empty(),goal)
        path_length = len(path)-1
        if path == []:
            result = 'no_path'
        elif path_length!=true_length: 
            result = 'wrong_path_size'
    if verbose and result!='success':
            print(f'{result} : ',node_name)
    return result,path_length


if __name__ == '__main__':
    block_sizes = [3,5]
    check_stack_number(block_sizes)
    test_eval_goals(block_sizes)
    # check_generate_goals()
