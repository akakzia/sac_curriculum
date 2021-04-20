from collections import defaultdict
import copy
import numpy as np
import random 
from curriculum.SemanticOperation import all_stack_trajectories


class TrajectoryGuidingSampler():
    
    def __init__(self,nb_blocks,target_stack):
        # probablement mieux de le mettre à l'extérieur de la classe. 
        self.goals_trajectory_dict = all_stack_trajectories(nb_blocks)
        # convert to GANGSTR boolean values : 

        self.target_stack = target_stack
        
    def sample_play_goal(self):
        goals = None
        if self.target_stack == None:
            target_stack = random.sample(self.goals_trajectory_dict.keys())
        else : 
            target_stack =self.target_stack
        goals = self.goals_trajectory_dict[target_stack]
        return copy.copy(goals)
        
    def generate_eval_goals(self):
        config_paths = list(self.goals_trajectory_dict.values())
        return copy.copy(config_paths)

    def evaluation(self,rollout_worker,eval_masks,max_traj=6):
        sr_results = defaultdict(list)

        target_trajectories = list(self.goals_trajectory_dict.items())[:max_traj]

        for target_stack,config_path in target_trajectories:
            # evaluation while teacher is guiding student : 
            episodes = rollout_worker.generate_rollout(goals=np.array(config_path),masks=eval_masks,self_eval=True,
                                                       true_eval=True,  biased_init=False,trajectory_goal = True)
            for i,episode in enumerate(episodes):
                sr = episode['success'][-1].astype(np.float32)
                sr_results[f"stack{i+2}_sr_guided"].append(sr)
            if target_stack == self.target_stack:
                sr = episodes[-1]['success'][-1].astype(np.float32)
                sr_results[f"stack_{target_stack}_guided"] = sr

            # evaluation only with goal : 
            goal = config_path[-1]
            episodes = rollout_worker.generate_rollout(goals=np.array([goal]),masks=eval_masks,self_eval=True,
                                                       true_eval=True,  biased_init=False,)
            sr = episodes[0]['success'][-1].astype(np.float32)
            sr_results[f"stack_sr_goal"].append(sr)
            if target_stack == self.target_stack:
                sr_results[f"stack_{target_stack}_goal"] = sr

        return sr_results
        


