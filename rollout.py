from graph.semantic_graph import SemanticGraph
import numpy as np
from language.build_dataset import sentence_from_configuration
from utils import language_to_id
from graph.SemanticOperation import SemanticOperation,config_to_unique_str

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


    def generate_rollout(self, goals, masks, self_eval, true_eval, biased_init=False, animated=False, language_goal=None):

        episodes = []
        for i in range(goals.shape[0]):
            observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=biased_init)
            obs = observation['observation']
            ag = observation['achieved_goal']
            ag_bin = observation['achieved_goal_binary']
            g = observation['desired_goal']
            g_bin = observation['desired_goal_binary']

            # in the language condition, we need to sample a language goal
            # here we sampled a configuration goal like in DECSTR, so we just use a language goal describing one of the predicates
            if self.args.algo == 'language':
                if language_goal is None:
                    language_goal_ep = sentence_from_configuration(g, eval=true_eval)
                else:
                    language_goal_ep = language_goal[i]
                lg_id = language_to_id[language_goal_ep]
            else:
                language_goal_ep = None
                lg_id = None

            ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []
            ep_lg_id = []
            ep_masks = []

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
                # Run policy for one step
                no_noise = self_eval or true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # feed both the observation and mask to the policy module
                action = self.policy.act(obs.copy(), ag.copy(), g.copy(), masks[i].copy(), no_noise, language_goal=language_goal_ep)

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
                ep_lg_id.append(lg_id)
                ep_success.append(is_success(ag_new, g, masks[i]))
                ep_masks.append(np.array(masks[i]).copy())

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
                           lg_ids=np.array(ep_lg_id).copy(),
                           masks=np.array(ep_masks).copy(),
                           self_eval=self_eval)

            if self.args.algo == 'language':
                episode['language_goal'] = language_goal_ep

            episodes.append(episode)

        return episodes


    def guided_rollouts(self, goals, self_eval, true_eval,semantic_graph:SemanticGraph,episode_duration,max_episodes=None,biased_init=False, animated=False):
        '''
            Runs rollout for each goals.
            For each goal, run pathfinding to find intermdiate goals.
            For each intermediate goal an epsiode is created. 
            if true_eval only return the last episode of each goals. 
        '''
        
        all_episodes = []
        end_goal_episode = []
        nb_episodes = 0 
        sem_op = SemanticOperation(5,True)
        class EpisodeBudgetReached(Exception): pass
        try : 
            for goal in goals:
                observation = self.env.unwrapped.reset_goal(goal=np.empty(0), biased_init=biased_init)
                self.last_obs = observation
                config_path = semantic_graph.get_path(observation['achieved_goal_binary'],goal)[1:]
                if len(config_path)==0:
                    config_path = [goal]
                for intermediate_goal in config_path:
                    goal_dist = len(semantic_graph.get_path_from_coplanar(self.last_obs["achieved_goal"])[1:])+1
                    episode = self.generate_one_rollout(intermediate_goal,goal_dist,self_eval, 
                                true_eval, episode_duration, animated=animated)
                    all_episodes.append(episode)
                    nb_episodes+=1
                    achieved_goal = episode['ag_binary'][-1]
                    
                    if max_episodes != None and nb_episodes >= max_episodes:
                        raise EpisodeBudgetReached
                    
                    if not (achieved_goal == intermediate_goal).all():
                        break
                end_goal_episode.append(episode)
        except EpisodeBudgetReached: 
            pass
        if true_eval:
            return end_goal_episode
        else : 
            return all_episodes

    def generate_one_rollout(self, goal,goal_dist, self_eval, true_eval, episode_duration, animated=False):            

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
            no_noise = self_eval or true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
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
                        #lg_ids=np.array([None]*len(ep_rewards)).copy(),
                        masks=np.array(ep_masks).copy(),
                        edge_dist=goal_dist,
                        self_eval=self_eval)

        self.last_obs = observation_new

        return episode