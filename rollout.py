import numpy as np

class RolloutWorker:
    def __init__(self, env, policy, goal_sampler, args):

        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.biased_init = args.biased_init
        self.goal_sampler = goal_sampler

    def generate_rollout(self, inits, goals, self_eval, true_eval, biased_init=False, animated=False):

        episodes = []
        for i in range(goals.shape[0]):

            observation = self.env.unwrapped.reset_goal(np.array(goals[i]), init=inits[i], biased_init=biased_init)

            obs = observation['observation']
            ag = observation['achieved_goal']
            ag_bin = observation['achieved_goal_binary']
            g = observation['desired_goal']
            g_bin = observation['desired_goal_binary']
            ep_obs, ep_ag, ep_ag_bin, ep_g_bin, ep_g, ep_actions, ep_success = [], [], [], [], [], [], []

            # start to collect samples
            for t in range(self.env_params['max_timesteps']):

                # run policy
                no_noise = self_eval or true_eval
                action = self.policy.act(obs.copy(), ag.copy(), g_bin.copy(), no_noise)

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                ag_new_bin = observation['achieved_goal_binary']

                # USE THIS FOR DEBUG
                # if str(ag_new) not in self.goal_sampler.valid_goals_str:
                #     # animated = True
                #     stop = 1

                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_ag_bin.append(ag_bin.copy())
                ep_g.append(g.copy())
                ep_g_bin.append(g_bin.copy())
                ep_actions.append(action.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_bin = ag_new_bin

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_ag_bin.append(ag_bin.copy())

            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           g_binary=np.array(ep_g_bin).copy(),
                           ag_binary=np.array(ep_ag_bin).copy(),
                           self_eval=self_eval)
            episodes.append(episode)


        return episodes

