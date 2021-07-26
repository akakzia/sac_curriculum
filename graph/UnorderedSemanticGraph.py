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

        new_mean_sr = self.edges_infos[edge]['SR']
        clamped_sr = max(np.finfo(float).eps, min(new_mean_sr, 1-np.finfo(float).eps))

        for n1,n2 in self.unordered_edge_to_ordered_edge[edge]:
            self.nk_graph.setWeight(n1,n2,-math.log(clamped_sr))

    def log(self,logger):
        logger.record_tabular('agent_nodes_ordered',self.nk_graph.numberOfNodes())
        logger.record_tabular('agent_edges_ordered',self.nk_graph.numberOfEdges())
        logger.record_tabular('agent_edges_unordered',len(self.edges_infos))

    def k_shortest_path(self,source, target,k,cutoff=10,use_weights=True,unordered_bias = True):
        '''
            Use Beam search combined with perfect path estimation to find k best paths. 
            if use_weights : use the edges weights, path is computed in amultiplicative way, highest score is best-score
            else :      each edges weigths is worth 1, path is computed in an additive way, smallest score is best-score
        '''
        if source == target : 
            return []
        
        reversed_sssp = self.get_sssp_to_goal(target,use_weight=use_weights) # sssp Single Source Shortest Path 
        target_node = self.configs[target]
        source_node = self.getNodeId(source)

        if use_weights:
            score_combination = lambda x,y : x*y
        else : 
            score_combination = lambda x,y : x+y

        k_cur_path_scores = [1 if use_weights else 0]
        k_best_path_nodes = np.array([[source_node]])
        k_best_path_finished = [False]
        
        for i in range(0,cutoff) : 
            next_paths_score_to_cur_node = []
            next_paths_score_to_goal = []
            next_paths_nodes = []
            next_path_finished = []

            # expand k best_path
            for cur_score,path,finished in zip(k_cur_path_scores,k_best_path_nodes,k_best_path_finished):    
                # get neighbors Scores :
                if not finished:
                    cur_node = path[i]
                    neighbors = list(self.nk_graph.iterNeighbors(cur_node))
                    for neigh in neighbors : 
                        if neigh in path [:i+1]:
                            continue
                        neigh_isgoal = (neigh == target_node)
                        path_to_goal,neigh_to_goal_sr,neigh_to_goal_dist = self.sample_shortest_path_with_sssp_from_nodes(neigh,target_node,reversed_sssp,return_configs=False,reversed=True)
                        if path_to_goal == None: 
                            continue
                        if use_weights : 
                            cur_to_neigh = np.exp(-self.getWeight_withNode(cur_node,neigh))
                            neigh_to_goal = neigh_to_goal_sr
                        else : 
                            cur_to_neigh = 1
                            neigh_to_goal = neigh_to_goal_dist
                        
                        score_to_neigh = score_combination(cur_score,cur_to_neigh)
                        score_to_goal = score_combination(score_to_neigh,neigh_to_goal)
                        if neigh_isgoal: 
                            full_path = np.concatenate((path[:i+1],np.array([neigh])))
                        else : 
                            full_path = np.concatenate((path[:i+1],np.array(path_to_goal)))
                        
                        if (len(full_path) < cutoff ) and (not use_weights or score_to_goal > 0):
                            next_paths_score_to_cur_node.append(score_to_neigh)
                            next_paths_score_to_goal.append(score_to_goal)
                            next_paths_nodes.append(full_path)
                            next_path_finished.append(neigh_isgoal)
                else : 
                    next_paths_score_to_cur_node.append(cur_score)
                    next_paths_nodes.append(path)
                    next_paths_score_to_goal.append(cur_score)
                    next_path_finished.append(finished)

            # filter similar paths 
            if unordered_bias: 
                next_paths_score_to_goal = np.array(next_paths_score_to_goal)
                inds = self.get_unique_unordered_paths(next_paths_nodes,next_paths_score_to_goal)
                next_paths_score_to_cur_node = [next_paths_score_to_cur_node[i] for i in inds]
                next_paths_score_to_goal = [next_paths_score_to_goal[i] for i in inds]
                next_paths_nodes = [next_paths_nodes[i] for i in inds]
                next_path_finished = [next_path_finished[i] for i in inds]
            
            # sort by scores and keep only k best : 
            if len(next_paths_score_to_cur_node)> k:
                next_paths_score_to_goal = np.array(next_paths_score_to_goal)
                if use_weights:
                    inds = np.argpartition(next_paths_score_to_goal, -k)[-k:]
                else : 
                    inds = np.argpartition(next_paths_score_to_goal, k)[:k]
                k_cur_path_scores = [next_paths_score_to_cur_node[i] for i in inds]
                k_best_path_nodes = [next_paths_nodes[i] for i in inds]
                k_best_path_finished = [next_path_finished[i] for i in inds]
            else : 
                k_cur_path_scores = next_paths_score_to_cur_node
                k_best_path_nodes = next_paths_nodes
                k_best_path_finished = next_path_finished
            
            if all(k_best_path_finished):
                break

        # sort k best paths before return : 
        k_cur_path_scores = np.array(k_cur_path_scores)
        order = -1 if use_weights else 1
        k_best_inds = np.argsort(order*k_cur_path_scores) # sort in correct order
        k_best_path_nodes = [k_best_path_nodes[i] for i in k_best_inds]
        k_best_path_configs = [list(map(lambda x: self.configs.inverse[x],path)) for path in k_best_path_nodes]
        k_cur_path_scores = k_cur_path_scores[k_best_inds]

        return k_best_path_configs,k_cur_path_scores

    def get_unique_unordered_paths(self,paths,scores):
        '''
            Receives an array of paths, paths are list of nodes. 
            Convert each path into a list of unordered edges, 
            If multiple paths are identical unordered-edes-wise, only return the id for one of them (at random among best scores).
        '''
        # create paths of unordered edges from paths of nodes
        unordered_edge_paths = np.ones((len(paths),max(map(len,paths))-1))*-1 # init with unused weight id
        for i,path in enumerate(paths):
            for j in range(0,len(path)-1):
                e = (path[j], path[j+1])
                unordered_edge_paths[i,j] = self.ordered_edge_to_unordered_edge[e]
        unique_unordered_paths,inverse_inds = np.unique(unordered_edge_paths,axis=0,return_inverse = True) 

        # sample among unordered edges paths with highest scores
        unique_ordered_paths_ids = np.zeros(unique_unordered_paths.shape[0],dtype=int)
        for i,e in enumerate(unique_unordered_paths):
            e_neigh_ids = np.where(inverse_inds == i)[0]
            e_scores = scores[e_neigh_ids]
            highest_scores = e_neigh_ids[np.argwhere(e_scores == np.amax(e_scores)).flatten()]
            choosen_id = np.random.choice(highest_scores)
            unique_ordered_paths_ids[i] = choosen_id

        return unique_ordered_paths_ids
