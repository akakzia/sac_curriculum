import random
from graph.semantic_graph import SemanticGraph
from mpi4py import MPI


class AgentNetwork():
    
    def __init__(self,semantic_graph :SemanticGraph,args):
        self.semantic_graph = semantic_graph
        self.frontier = [self.semantic_graph.empty()]
        self.reacheables_goals = set()
        self.args = args
        self.rank = MPI.COMM_WORLD.Get_rank()

    
    def sample_goal(self,k):
        if self.args.play_goal_strategy == 'uniform':
            return self.sample_goal_uniform(k)
        elif self.args.play_goal_strategy == 'frontier':
            return self.sample_goal_frontiere(k)

    def sample_goal_uniform(self,nb_goal):
        return random.choices(self.semantic_graph.configs.inverse,k=nb_goal) # sample with replacement

    def sample_goal_frontiere(self,nb_goal):
        return random.choices(self.frontier,k=nb_goal) # sample with replacement

    def update_frontiere(self,new_goal):
        goal_id = self.semantic_graph.getNode(new_goal)
        has_neighbours = False
        for neighbour_id in self.semantic_graph.nk_graph.iterNeighbors(goal_id):
            neighbour = self.semantic_graph.getConfig(neighbour_id)
            if neighbour not in self.reacheables_goals and neighbour not in self.frontier:
                has_neighbours = True
                self.frontier.append(neighbour)
        if has_neighbours and new_goal in self.frontier:
            self.frontier.remove(new_goal)

    def update(self,episodes):
        pass
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)
        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes 
                                    for e in eps] # flatten the list of episodes gathered by all actors
            for e in all_episode_list:
                start_config = e['ag'][0]
                achieved_goal = e['ag'][-1]
                goal = e['g'][-1]
                # goal is reacheable if the agent could reach it on purpose 
                if (achieved_goal == goal).all() and not (start_config == achieved_goal).all():
                    self.add_goal(achieved_goal)
        self.sync()
    
    def sync(self):
        self.frontier = MPI.COMM_WORLD.bcast(self.frontier, root=0)
        self.reacheables_goals = MPI.COMM_WORLD.bcast(self.reacheables_goals, root=0)


    def add_goal(self,goal):
        goal= tuple(goal)
        self.reacheables_goals.add(goal)
        self.update_frontiere(goal)

    def log(self,logger):
        logger.record_tabular('reachable_goals',len(self.reacheables_goals))
        logger.record_tabular('frontier_len',len(self.frontier))
