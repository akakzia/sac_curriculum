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
            observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=biased_init)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            g_desc = observation['goal_description']

            ep_obs, ep_ag, ep_g, ep_g_desc, ep_actions, ep_success = [], [], [], [], [], []

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
              # Run policy for one step
                no_noise = self_eval or true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # action = self.policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)
                action = self.policy.act_context(obs.copy(), g_desc.copy(), no_noise)

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                g_desc_new = observation['goal_description']

                # USE THIS FOR DEBUG
                # if str(ag_new) not in self.goal_sampler.valid_goals_str:
                #     # animated = True
                #     stop = 1

                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_g_desc.append(g_desc.copy())
                ep_actions.append(action.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new
                g_desc = g_desc_new

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g_desc.append(g_desc.copy())

            # Gather everything
            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           g_desc=np.array(ep_g_desc),
                           self_eval=self_eval)
            episodes.append(episode)

        return episodes

