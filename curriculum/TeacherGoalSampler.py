from collections import defaultdict
import copy
import numpy as np
import random 
from curriculum.SemanticOperation import all_stack_trajectories
from curriculum.trajectory_analysis import episode_trajectory_type

class TrajectoryGuidingSampler():
    
    def __init__(self,nb_blocks,target_stack):
        # probablement mieux de le mettre à l'extérieur de la classe. 
        self.config_size = nb_blocks * (nb_blocks - 1) * 3 // 2
        self.goals_trajectory_dict = all_stack_trajectories(nb_blocks)
        # convert to GANGSTR boolean values : 

        self.target_stack = target_stack
        self.nb_block = nb_blocks
        
    def sample_play_goal(self):
        goals = None
        if self.target_stack == None:
            target_stack = random.sample(self.goals_trajectory_dict.keys(),1)[0]
        else : 
            target_stack =self.target_stack
        goals = self.goals_trajectory_dict[target_stack]
        return copy.copy(goals)

    def evaluation(self,rollout_worker,max_traj=6,animated=False,verbose=False):
        sr_results = defaultdict(list)
        for trajectory_type in ['advised','custom','failure'] :
            for k in range(3,self.nb_block+1):
                sr_results[f"stack{k}_sr_goal_{trajectory_type}"]=0
        
        target_trajectories = random.sample(list(self.goals_trajectory_dict.items()),max_traj)
        for target_stack,config_path in target_trajectories:
            if verbose : 
                print("guided episodes : ",target_stack)
            config_path = copy.copy(config_path)
            # evaluation while teacher is guiding student : 
            episode_guided,nb_goal_reached = rollout_worker.guided_rollout(goals=np.array(config_path),self_eval=True,
                                                       true_eval=True,  biased_init=False, animated=animated,
                                                       consecutive_success=5)
            
            for i in range(nb_goal_reached):
                sr_results[f"stack{i+2}_sr_guided"].append(1)
            for i in range(nb_goal_reached,len(config_path)):
                sr_results[f"stack{i+2}_sr_guided"].append(0)
            
            # if target_stack == self.target_stack:
            #     sr_results[f"stack_target_guided"] = sr_results[f"stack{len(target_stack)}_sr_guided"]

            # evaluation only with goal : 
            if verbose : 
                print("goal only episodes : ",target_stack)
            goals = config_path[1:]
            eval_masks = np.array(np.zeros((len(config_path),self.config_size)))
            episodes_goal = rollout_worker.generate_rollout(goals=np.array(goals),masks=eval_masks,self_eval=True,
                                                       true_eval=True,  biased_init=False,
                                                       animated=animated)
            for i,episode in enumerate(episodes_goal):
                stack_size = i+3
                sr = episode['success'][-1].astype(np.float32)
                sr_results[f"stack{stack_size}_sr_goal"].append(sr)
                trajectory_type = episode_trajectory_type(episode,config_path[:stack_size])
                sr_results[f"stack{stack_size}_sr_goal_{trajectory_type}"]+=1

            # if target_stack == self.target_stack:
            #     sr_results[f"stacktarget_goal"] = sr
            
        return sr_results