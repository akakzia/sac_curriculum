from graph.semantic_graph import SemanticGraph
from mpi4py import MPI
from graph.teacher import Teacher
import pickle

class AgentNetwork():
    
    def __init__(self,semantic_graph :SemanticGraph,args):
        self.teacher = Teacher(args)
        self.semantic_graph = semantic_graph
        self.args = args
        self.rank = MPI.COMM_WORLD.Get_rank()

    def update(self,episodes):
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)
        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes 
                                    for e in eps] # flatten the list of episodes gathered by all actors
            # update agent graph : 
            for e in all_episode_list:
                start_config = tuple(e['ag'][0])
                achieved_goal = tuple(e['ag'][-1])
                goal = tuple(e['g'][-1])
                success = e['success'][-1]
                
                self.semantic_graph.add_config(start_config)
                self.semantic_graph.add_config(achieved_goal)
                self.semantic_graph.add_config(goal)
                self.update_edge(start_config,goal,success)
                self.update_edge(start_config,achieved_goal,True)

            # update frontier :  
            self.teacher.computeFrontier(self.semantic_graph)
            self.semantic_graph.update_shortest_tree()
        self.sync()
    
    def update_edge(self,start,end,success):
        if (start!=end):
            self.semantic_graph.update_edge((start,end),success)
    
    def get_path(self,start,goal):
        if self.args.expert_graph_start: 
            return self.teacher.oracle_graph.get_path(start,goal)
        else : 
            return self.semantic_graph.get_path(start,goal)

    def get_path_from_coplanar(self,target):
        if self.args.expert_graph_start : 
            return self.teacher.oracle_graph.get_path_from_coplanar(target)
        else : 
            return self.semantic_graph.get_path_from_coplanar(target)

    def sample_goal(self,k):
        if self.args.play_goal_strategy == 'uniform':
            return self.sample_goal_uniform(k)
        elif self.args.play_goal_strategy == 'frontier':
            return self.sample_goal_in_frontier(k)

    def sample_goal_uniform(self,nb_goal):
        return self.teacher.sample_goal_uniform(nb_goal)

    def sample_goal_in_frontier(self,k):
        return self.teacher.sample_in_frontier(k)
    
    def sample_from_frontier(self,frontier_node,k):
        return self.teacher.sample_from_frontier(frontier_node,self.semantic_graph,k)

    def log(self,logger):
        logger.record_tabular('agent_nodes',self.semantic_graph.nk_graph.numberOfNodes())
        logger.record_tabular('agent_edges',self.semantic_graph.nk_graph.numberOfEdges())
        logger.record_tabular('frontier_len',len(self.teacher.agent_frontier))

    def save(self,model_path, epoch):
        self.semantic_graph.save(model_path+'/',f'{epoch}')
        with open(f"{model_path}/frontier_{epoch}.config", 'wb') as f:
            pickle.dump(self.teacher.agent_frontier,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(model_path,epoch,args) ->'AgentNetwork':
        semantic_graph = SemanticGraph.load(model_path,f'{epoch}',args.n_blocks)
        with open(f"{model_path}frontier_{epoch}.config", 'rb') as f:
            frontier = pickle.load(f)
        agent_network = AgentNetwork(semantic_graph,args)
        agent_network.teacher.agent_frontier = frontier
        return agent_network

    def sync(self):
        self.teacher.agent_frontier = MPI.COMM_WORLD.bcast(self.teacher.agent_frontier, root=0)
        if self.rank == 0:
            self.semantic_graph.save('data/','temp')
        MPI.COMM_WORLD.Barrier()
        if self.rank!=0:
            self.semantic_graph = SemanticGraph.load('data/','temp',self.args.n_blocks)