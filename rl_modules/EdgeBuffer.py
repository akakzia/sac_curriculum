
from collections import defaultdict
import threading
import numpy as np
import random
from graph.SemanticOperation import SemanticOperation,config_to_unique_str

class EdgeBuffer:
    def __init__(self, env_params, buffer_size, sample_func, replay_sampling,args):
        self.env_params = env_params
        self.T = args.episode_duration
        self.max_size = buffer_size // self.T
        self.replay_sampling = replay_sampling

        self.sample_func = sample_func

        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.max_size, self.T + 1, self.env_params['obs']]),
                       'ag': np.empty([self.max_size, self.T + 1, self.env_params['goal']]),
                       'g': np.empty([self.max_size, self.T, self.env_params['goal']]),
                       'actions': np.empty([self.max_size, self.T, self.env_params['action']]),
                       }
        self.edges_to_infos = defaultdict(lambda : {'edge_dist':None,'episode_ids':[]}) # edges : {dist:2, episode_ids:[ 0,12]}
        self.all_edges = []
        # thread lock
        self.lock = threading.Lock()


    # store the episode
    def store_episodes(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            for i, e in enumerate(episode_batch):
                self.store_episode(e,idxs[i])
            self.all_edges = list(self.edges_to_infos)   # use separate buffer for edge sampling

    def store_episode(self,episode,identifiant):
        # update edge infos in buffer :
        edge = (tuple(episode['ag'][0]),tuple(episode['ag'][-1]))
        self.edges_to_infos[edge]['episode_ids'].append(identifiant)
        self.edges_to_infos[edge]['edge_dist'] = episode['edge_dist']
        test = self.buffer['ag'][identifiant]==None
        if self.buffer['ag'][identifiant] is not None:
            last_stored_edge = (tuple(self.buffer['ag'][identifiant][0]),tuple(self.buffer['ag'][identifiant][-1]))
            if identifiant in self.edges_to_infos[last_stored_edge] : 
                self.edges_to_infos[edge]['episode_ids'].remove(identifiant)
            if len(self.edges_to_infos[last_stored_edge]['episode_ids']) == 0:
                del self.edges_to_infos[last_stored_edge]
        # store the episode in buffer
        self.buffer['obs'][identifiant] = episode['obs']
        self.buffer['ag'][identifiant] = episode['ag']
        self.buffer['g'][identifiant] = episode['g']
        self.buffer['actions'][identifiant] = episode['act']
        if 'language_goal' in episode.keys():
            self.buffer['lg_ids'][identifiant] = episode['lg_ids']


    def sample_edge(self,batch_size):
        return random.choices(self.all_edges,k=batch_size)

    def distance_biased_sample_edge(self,batch_size):
        edges,edges_infos = zip(*self.edges_to_infos.items())
        distances = [edges_info['edge_dist'] for edges_info in edges_infos]
        return random.choices(edges,distances,k=batch_size)
    
    def sample_transition(self,batch_size):
        temp_buffers = {}
        with self.lock:
            if self.replay_sampling == 'buffer_uniform':
                ep_ids = np.random.randint(0,self.current_size,size=batch_size)
            else:
                if self.replay_sampling == 'edge_uniform':
                    edges = self.sample_edge(batch_size)
                elif self.replay_sampling == 'edge_distance':
                    edges = self.distance_biased_sample_edge(batch_size)
                else:
                    raise Exception('unknow replay method')
                ep_ids = np.zeros(batch_size,dtype=int)
                for i,edge in enumerate(edges):
                    ep_ids[i] = np.random.choice(self.edges_to_infos[edge]['episode_ids'])
            for key in self.buffer.keys():
                temp_buffers[key] = self.buffer[key][ep_ids]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

         # HER Re-Labelling : 
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def sample(self,batch_size):
        return self.sample_transition(batch_size)
        
    def get_nb_edges(self):
        return len(self.edges_to_infos)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.max_size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.max_size:
            overflow = inc - (self.max_size - self.current_size)
            idx_a = np.arange(self.current_size, self.max_size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.max_size, inc)
        self.current_size = min(self.max_size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx


class EdgeBufferDynEpisodes(EdgeBuffer):
    '''
        Replay buffer containing episodes with differents lengths. 
    '''

    def __init__(self, env_params, buffer_size, sample_func, replay_sampling):
        self.env_params = env_params
        self.max_transitions = buffer_size
        self.replay_sampling = replay_sampling
        self.sample_func = sample_func
        self.current_transitions = 0

        self.buffer = {'obs': [],'ag': [],'g': [],'actions': []}
        self.edges_to_infos = defaultdict(lambda : {'edge_dist':None,'episode_ids':[]}) # edges : {dist:2, episode_ids:[ 0,12]}
        self.all_edges = []
        # thread lock
        self.lock = threading.Lock()
            
        # store the episode
    def store_episodes(self, episode_batch):
        with self.lock:
            idxs = self._get_storage_idx(list(map(lambda e : len(e['obs']),episode_batch)))
            for i, e in enumerate(episode_batch):
                self.store_episode(e,idxs[i])
            self.all_edges = list(self.edges_to_infos)   # use separate buffer for edge sampling
            self.current_transitions = sum(map(len,self.buffer['obs']))

    def _get_storage_idx(self, episodes_len):
        nb_new_transitions = np.sum(episodes_len)
        nb_episodes = len(episodes_len)

        if self.current_transitions + nb_new_transitions <= self.max_transitions:
            idx = np.arange(len(self.buffer['obs']), len(self.buffer['obs']) + nb_episodes)
            for _ in range(nb_episodes):
                for key in self.buffer.keys():
                    self.buffer[key].append(None)
        else:
            idx = np.random.randint(0, len(self.buffer), nb_episodes)
        return idx

    def sample_transition(self,batch_size):
        temp_buffers = defaultdict(lambda : [None]*batch_size)
        with self.lock:
            if self.replay_sampling == 'buffer_uniform':
                ep_ids = np.random.randint(0,self.current_size,size=batch_size)
            else:
                if self.replay_sampling == 'edge_uniform':
                    edges = self.sample_edge(batch_size)
                elif self.replay_sampling == 'edge_distance':
                    edges = self.distance_biased_sample_edge(batch_size)
                else:
                    raise Exception('unknow replay method')
                ep_ids = np.zeros(batch_size,dtype=int)
                for i,edge in enumerate(edges):
                    ep_ids[i] = np.random.choice(self.edges_to_infos[edge]['episode_ids'])

            for key in self.buffer.keys():
                for i,e_id in enumerate(ep_ids):
                    temp_buffers[key][i] = self.buffer[key][e_id]
                    if key == 'obs':
                        temp_buffers['obs_next'][i] = self.buffer[key][e_id][1:]
                    elif key == 'ag':
                        temp_buffers['ag_next'][i] = self.buffer[key][e_id][1:]

        # HER Re-Labelling : 
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions
