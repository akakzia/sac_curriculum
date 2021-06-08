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


    def train_rollout(self,agentNetwork:AgentNetwork,episode_duration,max_episodes=None,time_dict=None, animated=False):
        all_episodes = []

        while len(all_episodes) < max_episodes:
            # step 1 : go to the frontier of knowns goals
            t_i = time.time()
            goal = agentNetwork.sample_goal(1)[0]
            if time_dict:
                time_dict['goal_sampler'] += time.time() - t_i
            episodes,_ = self.guided_rollout(goal,False, agentNetwork, episode_duration, 
                                            episode_budget=max_episodes-len(all_episodes), biased_init=False,animated=animated)
            all_episodes+= episodes
            # step 2 : explore outside the frontier of known goals
            last_ag = episodes[-1]['ag'][-1]
            if (self.args.play_goal_strategy == 'frontier' and len(all_episodes) < max_episodes 
                and (last_ag == goal).all()):
                t_i = time.time()
                frontier_goal = next(iter(agentNetwork.sample_from_frontier(goal,1)),None) # first element or None
                if frontier_goal:
                    goal_dist = episodes[-1]["edge_dist"]+1
                    if time_dict:
                        time_dict['goal_sampler'] += time.time() - t_i
                    episode = self.generate_one_rollout(frontier_goal, goal_dist, False, episode_duration,animated=animated)
                    all_episodes.append(episode)
            
        return all_episodes
    
    def test_rollout(self,goals,agent_network:AgentNetwork,episode_duration, animated=False):
        end_episodes = []
        for goal in goals : 
            _,last_episode = self.guided_rollout(goal,True, agent_network, episode_duration, biased_init=False, animated=animated)
            end_episodes.append(last_episode)
        return end_episodes


    def guided_rollout(self,goal,evaluation,agent_network:AgentNetwork,episode_duration,episode_budget=None,biased_init=False, animated=False):
        episodes = []
        observation = self.env.unwrapped.reset_goal(goal=np.array(goal), biased_init=biased_init)
        self.last_obs = observation
        config_path = agent_network.get_path(observation['achieved_goal_binary'],goal)[1:]
        sem_op = SemanticOperation(5,True)
        # print(f'goal : {config_to_name(goal,sem_op)}')
        # print("path : ",[config_to_name(c,sem_op) for c in config_path])

        if len(config_path)==0:
            config_path = [goal]
        for intermediate_goal in config_path:
            # print("intermediate : ",config_to_name(intermediate_goal,sem_op))
            goal_dist = len(agent_network.get_path_from_coplanar(self.last_obs["achieved_goal"])[1:])+1
            episode = self.generate_one_rollout(intermediate_goal,goal_dist, 
                                                evaluation, episode_duration, animated=animated)
            episodes.append(episode)
            achieved_goal = episode['ag_binary'][-1]
            
            if episode_budget != None and len(episodes) >= episode_budget:
                break
            
            if not (achieved_goal == intermediate_goal).all():
                break            
        
        last_episode = episode
        return episodes,last_episode

    def generate_one_rollout(self, goal,goal_dist, evaluation, episode_duration, animated=False):            

        g = np.array(goal)
        self.env.unwrapped.target_goal = np.array(goal)
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

        return episode