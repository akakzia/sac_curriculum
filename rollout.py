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
        self.dijkstra_to_goal = None
        self.state ='GoToFrontier'

    def train_rollout(self,agentNetwork:AgentNetwork,episode_duration,max_episodes=None,time_dict=None, animated=False,biased_init=False):
        all_episodes = []

        while len(all_episodes) < max_episodes:
            
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
                if self.args.edge_exploration:
                    episodes,_ = self.explore_toward_goal(self.long_term_goal, agentNetwork, episode_duration, 
                                        episode_budget=max_episodes-len(all_episodes),animated=animated)
                else :
                    episodes,_ = self.guided_rollout(self.long_term_goal,False, agentNetwork, episode_duration, 
                                            episode_budget=max_episodes-len(all_episodes),animated=animated)
                all_episodes += episodes

                success = episodes[-1]['success'][-1]
                if success == False: # reset at the first failure
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
                    episode = self.generate_one_rollout(explore_goal, goal_dist, False, episode_duration,animated=animated)
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
            _,last_episode = self.guided_rollout(goal,True, agent_network, episode_duration, animated=animated)
            end_episodes.append(last_episode)
        self.reset(False)
        return end_episodes

    def explore_toward_goal(self,goal,agent_network:AgentNetwork,episode_duration,episode_budget=None, animated=False):
        episode = None
        episodes = []
        goal = tuple(goal)
        sem_op = SemanticOperation(5,True)

        if animated : 
            print('goal : ',config_to_unique_str(goal,sem_op))

        if self.dijkstra_to_goal == None:
            self.current_goal_id = 1
            self.dijkstra_to_goal = agent_network.semantic_graph.get_dijkstra_to_goal(goal)
            
        while self.current_config != goal:
            goal_dist = self.current_goal_id
            current_goal = None
            if self.dijkstra_to_goal: 
                current_goal = agent_network.sample_neighbour_based_on_SR_to_goal(self.current_config,self.dijkstra_to_goal,goal=goal)
            if current_goal == None: # if no path to goal, try to reach directly 
                current_goal = goal
            if animated:
                print(f'\t{self.current_goal_id} : \n{config_to_unique_str(self.current_config,sem_op)} ->  {config_to_unique_str(current_goal,sem_op)}'  )

            episode = self.generate_one_rollout(current_goal,goal_dist, 
                                                False, episode_duration, animated=animated)
            episodes.append(episode)
            self.current_goal_id+=1
            success = episodes[-1]['success'][-1]

            if episode_budget != None and len(episodes) >= episode_budget:
                break

            if success == False:
                if animated : 
                    print("failure")
                break 
            elif animated : 
                print('success')

        return episodes,self.last_episode



    def guided_rollout(self,goal,evaluation,agent_network:AgentNetwork,episode_duration,episode_budget=None, animated=False):
        episode = None
        episodes = []
        goal = tuple(goal)
        sem_op = SemanticOperation(5,True)

        if animated : 
            print('goal : ',config_to_unique_str(goal,sem_op))

        if self.current_goal_id == None:
            self.current_goal_id = 1
            self.config_path,_ = agent_network.get_path(self.current_config,goal)
            if len(self.config_path)==0:
                self.config_path = [self.current_config,goal]

        while self.current_goal_id < len(self.config_path):
            goal_dist = self.current_goal_id
            current_goal = self.config_path[self.current_goal_id]
            
            if animated:
                print(f'\t{self.current_goal_id} : \n{config_to_unique_str(self.current_config,sem_op)} ->  {config_to_unique_str(current_goal,sem_op)}'  )
                
            episode = self.generate_one_rollout(current_goal,goal_dist, 
                                                evaluation, episode_duration, animated=animated)
            episodes.append(episode)
            self.current_goal_id+=1
            
            success = episodes[-1]['success'][-1]

            if episode_budget != None and len(episodes) >= episode_budget:
                break

            if success == False:
                break 

        return episodes,self.last_episode

    def generate_one_rollout(self, goal,goal_dist, evaluation, episode_duration, animated=False):            

        g = np.array(goal)
        self.env.unwrapped.target_goal = np.array(goal)
        self.env.unwrapped.binary_goal = np.array(goal)
        obs = self.last_obs['observation']
        ag = self.last_obs['achieved_goal']
        ag_bin = self.last_obs['achieved_goal_binary']
        g_bin = self.last_obs['desired_goal_binary']
        empty_mask = np.zeros(len(goal))

        ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []
        ep_masks = []
        # Start to collect samples
        for _ in range(episode_duration):
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

        self.last_obs = observation_new
        self.last_episode = episode

        return episode