import random
from graph.agent_network import AgentNetwork
import numpy as np
from graph.SemanticOperation import SemanticOperation, config_to_name,config_to_unique_str
import time 
def is_success(ag, g, mask=None):
    if mask is None:
        return (ag == g).all()
    else:
        ids = np.where(mask != 1.)[0]
        return (ag[ids] == g[ids]).all()


class RolloutWorker:
    def __init__(self, env, policy, goal_sampler, args):
        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.biased_init = args.biased_init
        self.goal_sampler = goal_sampler
        self.args = args
        self.last_obs = None
        self.total_train_cycle_steps = 0
        self.total_rollout_episodes = 0
        self.reset(False)

    @property
    def current_config(self):
        return tuple(self.last_obs['achieved_goal_binary'])

    def reset(self,biased_init):
        self.long_term_goal = None
        self.config_path = None
        self.current_goal_id = None
        self.last_episode = None
        self.last_obs = self.env.unwrapped.reset_goal(goal=np.array([None]), biased_init=biased_init)
        self.total_steps_to_goal = 0
        self.dijkstra_to_goal = None
        self.state ='GoToFrontier'

    def train_rollout(self,agentNetwork:AgentNetwork,time_dict=None, animated=False,biased_init=False):
        all_episodes = []

        self.total_train_cycle_steps = 0
        self.total_rollout_episodes = 0
        while not self.train_end_criteria():
            
            if self.state == 'GoToFrontier':
                if self.long_term_goal == None : 
                    t_i = time.time()
                    self.long_term_goal = next(iter(agentNetwork.sample_goal(self.current_config,1)),None) # first element or None
                    if time_dict:
                        time_dict['goal_sampler'] += time.time() - t_i
                    # if can't find frontier goal, explore directly
                    if self.long_term_goal == None or self.long_term_goal == self.current_config: 
                        self.state = 'Explore'
                        continue
                episodes,_ = self.guided_rollout(self.long_term_goal,False, agentNetwork, animated=animated)
                all_episodes += episodes

                success = episodes[-1]['success'][-1]
                if self.guided_rollout_end_criteria(success):
                    self.reset(biased_init)
                elif success and self.current_config == self.long_term_goal:
                    self.state = 'Explore'

            elif self.state =='Explore':
                t_i = time.time()
                last_ag = tuple(self.last_obs['achieved_goal_binary'])
                explore_goal = next(iter(agentNetwork.sample_from_frontier(last_ag,1)),None) # first element or None
                if time_dict:
                    time_dict['goal_sampler'] += time.time() - t_i
                if explore_goal:
                    if self.last_episode:
                        goal_dist = self.last_episode["edge_dist"]+1
                    else : 
                        goal_dist = 1
                    episode = self.generate_one_rollout(explore_goal, goal_dist, False,animated=animated)
                    all_episodes.append(episode)
                    success = episode['success'][-1]
                if explore_goal == None or  success == False:
                        self.reset(biased_init)
                        continue
            else : 
                raise Exception(f"unknown state : {self.state}")
        return all_episodes
    
    
    def test_rollout(self,goals,agent_network:AgentNetwork,episode_duration, animated=False):
        end_episodes = []
        for goal in goals : 
            self.reset(False)
            _,last_episode = self.guided_rollout(goal,True, agent_network, animated=animated)
            end_episodes.append(last_episode)
        self.reset(False)
        return end_episodes

    def plan(self,agent_network,goal,evaluation):
        if self.args.rollout_exploration =='sr_and_k_distance':
            self.current_goal_id = 1
            if evaluation or np.random.rand()< self.args.rollout_distance_ratio:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal)
            else : 
                k_best_paths,_ = agent_network.semantic_graph.k_shortest_path(self.current_config,goal,
                                                                                        self.args.rollout_exploration_k,
                                                                                        use_weights = False,
                                                                                        unordered_bias = True)
                self.config_path = random.choices(k_best_paths,k=1)[0] if k_best_paths else None
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.current_goal_id = 1
            if evaluation or np.random.rand()< self.args.rollout_distance_ratio:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='dijkstra')
            else : 
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='bfs')
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration=='sample_sr' :
            self.current_goal_id = 1
            self.dijkstra_to_goal = agent_network.semantic_graph.get_sssp_to_goal(goal)
        else : raise Exception('unknown exploration method') 
        
    def get_next_goal(self,agent_network,goal,evaluation):
        
        if self.args.rollout_exploration =='sr_and_k_distance':
            if self.current_goal_id == None or self.config_path[self.current_goal_id-1] != self.current_config:
                self.plan(agent_network,goal,evaluation)
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.plan(agent_network,goal,evaluation)
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration=='sample_sr' :
            current_goal = None
            if self.dijkstra_to_goal: 
                current_goal = agent_network.sample_neighbour_based_on_SR_to_goal(self.current_config,self.dijkstra_to_goal,goal=goal)
            if current_goal == None: # if no path to goal, try to reach directly 
                current_goal = goal
        else : raise Exception('unknown exploration method') 

        return current_goal,self.current_goal_id

    def guided_rollout(self,goal,evaluation,agent_network:AgentNetwork, animated=False):
        episode = None
        episodes = []
        goal = tuple(goal)
        sem_op = SemanticOperation(5,True)

        if animated : 
            print('goal : ',config_to_unique_str(goal,sem_op))
        if self.current_goal_id == None:
            self.plan(agent_network,goal,evaluation)

        while True:
            current_goal,goal_dist = self.get_next_goal(agent_network,goal,evaluation)
            
            if animated:
                print(f'\t{self.current_goal_id} : \n{config_to_unique_str(self.current_config,sem_op)} ->  {config_to_unique_str(current_goal,sem_op)}'  )
                
            episode = self.generate_one_rollout(current_goal,goal_dist, 
                                                evaluation, animated=animated)
            episodes.append(episode)
            self.current_goal_id+=1
            success = episodes[-1]['success'][-1]

            if self.guided_rollout_break_criteria(evaluation,success):
                break
            if current_goal == goal:
                break

        return episodes,self.last_episode

    def generate_one_rollout(self, goal,goal_dist, evaluation, animated=False):            

        g = np.array(goal)
        self.env.unwrapped.target_goal = np.array(goal)
        self.env.unwrapped.binary_goal = np.array(goal)
        obs = self.last_obs['observation']
        ag = self.last_obs['achieved_goal']
        ag_bin = self.last_obs['achieved_goal_binary']
        g_bin = self.last_obs['desired_goal_binary']
        empty_mask = np.zeros(len(goal))
        start_config = tuple(ag_bin)

        if goal == None:
            raise Exception("not supposed to happen")

        ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []
        ep_masks = []
        # Start to collect samples
        while (True):
            # Run policy for one step
            no_noise = evaluation  # do not use exploration noise if running self-evaluations or offline evaluations
            # feed both the observation and mask to the policy module
            action = self.policy.act(obs.copy(), ag.copy(), g.copy(), empty_mask.copy(), no_noise, language_goal=False)

            # feed the actions into the environment
            if animated:
                self.env.render()

            observation_new, r, _, _ = self.env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            ag_new_bin = observation_new['achieved_goal_binary']

            # Append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_ag_bin.append(ag_bin.copy())
            ep_g.append(g.copy())
            ep_g_bin.append(g_bin.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(r)
            ep_success.append(is_success(ag_new, g, empty_mask))
            ep_masks.append(np.array(empty_mask).copy())

            # Re-assign the observation
            obs = obs_new
            ag = ag_new
            ag_bin = ag_new_bin
            
            self.total_steps_to_goal+=1
            self.total_train_cycle_steps+=1
            if self.episode_end_criteria(start_config,tuple(ag_new),len(ep_obs)):
                break

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_ag_bin.append(ag_bin.copy())

        # Gather everything
        episode = dict(obs=np.array(ep_obs).copy(),
                        act=np.array(ep_actions).copy(),
                        g=np.array(ep_g).copy(),
                        ag=np.array(ep_ag).copy(),
                        success=np.array(ep_success).copy(),
                        g_binary=np.array(ep_g_bin).copy(),
                        ag_binary=np.array(ep_ag_bin).copy(),
                        rewards=np.array(ep_rewards).copy(),
                        masks=np.array(ep_masks).copy(),
                        edge_dist=goal_dist,
                        self_eval=evaluation)

        if animated : 
            print('success :', ep_success[-1],)

        self.last_obs = observation_new
        self.last_episode = episode
        self.total_rollout_episodes+=1

        return episode

    def episode_end_criteria(self,start_config,current_config,ep_len):
        return ep_len >= self.args.episode_duration

    def guided_rollout_break_criteria(self,evaluation,success):
        if evaluation:
            return self.guided_rollout_end_criteria(success)
        else : 
            return (self.train_end_criteria()
                    or self.guided_rollout_end_criteria(success))

    def guided_rollout_end_criteria(self,success):
        return not success

    def train_end_criteria(self):
        return self.total_rollout_episodes >= self.args.num_rollouts_per_mpi

class RolloutWorkerStepBudget(RolloutWorker):
    '''
    This worker has a step budget : episode_duration * episode_budget.
    And a goal budget : episode_duration
    The length of an intermediate episode to reach a intermediate goal can vary (at most episode_duration steps). 
    '''

    def episode_end_criteria(self,start_config,current_config,ep_len):
        return start_config != current_config or self.total_steps_to_goal > self.args.episode_duration
 
    def guided_rollout_end_criteria(self,success):
        return self.total_steps_to_goal >= self.args.episode_duration

    def train_end_criteria(self):
        return self.total_train_cycle_steps >= self.args.episode_duration * self.args.num_rollouts_per_mpi