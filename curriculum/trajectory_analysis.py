
def check_order(achieved_goals,target_trajectory):
    ''' Returns true if an achieved goal list passes through every intermediate goals given in 
        a target trajectory '''
    intermediate_goal_id = 0
    for ag in achieved_goals:
        if tuple(ag) == target_trajectory[intermediate_goal_id]:
            intermediate_goal_id+=1
            if intermediate_goal_id == len(target_trajectory):
                return True
    return False
    
def episode_trajectory_type(episode,target_trajectory):
    ''' Classify episodes in 3 categories : 
        - failure : if the episode succès is false in the last state
        - advised : if the episode ends in success while following the guided trajectory
        - custom : if the episode ends in success while following any other trajectory
    '''
    if episode['success'][-1] == 0 : 
        return 'failure'
    elif check_order(episode['ag_binary'],target_trajectory):
        return 'advised'
    else :
        return 'custom'