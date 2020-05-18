import numpy as np
import os
import itertools

from env import rotations, robot_env, utils


def objects_distance(x, y):
    """
    A function that returns the euclidean distance between two objects x and y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x - y)


def above(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x[:2] - y[:2]) < 0.07 and 0.055 > x[2] - y[2] > 0.03
    # return np.linalg.norm(x[:2] - y[:2]) < 0.07 and 0.06 > np.abs(x[2] - y[2]) > 0.01


class FetchManipulateEnvContinuous(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, num_blocks, n_substeps, gripper_extra_height, block_gripper,
        target_in_the_air, target_offset, obj_range, target_range, predicate_threshold,
            initial_qpos, reward_type, predicates,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            num_blocks: number of block objects in environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        
        self.target_goal = None
        self.binary_goal = None
        self.num_blocks = num_blocks
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.predicate_threshold = predicate_threshold
        self.predicates = predicates
        self.num_predicates = len(self.predicates)
        self.guide = False
        self.reward_type = reward_type

        self.goal_size = 0

        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        self.location_record = None
        self.location_record_write_dir = None
        self.location_record_prefix = None
        self.location_record_file_number = 0
        self.location_record_steps_recorded = 0
        self.location_record_max_steps = 2000


        # temporary values
        self.min_max_x=(1.19,1.49)
        self.min_max_y = (0.59, 0.89)
        super(FetchManipulateEnvContinuous, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        self.min_max_x = (self.initial_gripper_xpos[0] - self.obj_range, self.initial_gripper_xpos[0] + self.obj_range)
        self.min_max_y = (self.initial_gripper_xpos[1] - self.obj_range, self.initial_gripper_xpos[1] + self.obj_range)


    # Heatmap Generation
    # ----------------------------

    def set_location_record_name(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_location_record()
        # if not self.record_write_dir:
        #     self.save_heatmap_picture(os.path.join(write_dir,'level.png'))
        self.location_record_write_dir = write_dir
        self.location_record_prefix = prefix
        self.location_record_file_number = 0

        return True

    def flush_location_record(self, create_new_empty_record=True):
        if self.location_record is not None and self.location_record_steps_recorded > 0:
            write_file = os.path.join(self.location_record_write_dir,"{}_{}".format(self.location_record_prefix,
                                                                           self.location_record_file_number))
            np.save(write_file, self.location_record[:self.location_record_steps_recorded])
            self.location_record_file_number += 1
            self.location_record_steps_recorded = 0

        if create_new_empty_record:
            self.location_record = np.empty(shape=(self.location_record_max_steps, 3), dtype=np.float32)

    def log_location(self, location):
        if self.location_record is not None:
            self.location_record[self.location_record_steps_recorded] = location
            self.location_record_steps_recorded += 1

            if self.location_record_steps_recorded >= self.location_record_max_steps:
                self.flush_location_record()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        assert self.reward_type in ['sparse', 'incremental']
        if self.reward_type == 'incremental':
            # Using incremental reward for each correct predicate
            raise NotImplementedError
            # reward = np.sum(achieved_goal == goal).astype(np.float32) - self.num_blocks
        else:
            # Using sparse reward
            reward = (achieved_goal == goal).all().astype(np.float32)
        return reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        self.log_location(location=self.sim.data.get_site_xpos('robot0:grip'))
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
    
    def _get_configuration(self, positions):
        close_config = np.array([])
        above_config = np.array([])
        if "close" in self.predicates:
            object_combinations = itertools.combinations(positions, 2)
            object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

            close_config = np.array([(distance <= self.predicate_threshold).astype(np.float32)
                                     for distance in object_rel_distances])
        if "above" in self.predicates:
            if self.num_blocks == 3:
                object_permutations = [(positions[0], positions[1]), (positions[1], positions[0]), (positions[0], positions[2]),
                                       (positions[2], positions[0]), (positions[1], positions[2]), (positions[2], positions[1])]
            else:
                raise NotImplementedError

            above_config = np.array([int(above(obj[0], obj[1])) for obj in object_permutations]).astype(np.float32)
        
        res = np.concatenate([close_config, above_config])
        return res

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        objects_positions = []

        for i in range(self.num_blocks):

            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            objects_positions = np.concatenate([objects_positions, object_i_pos.ravel()])

        objects_positions = objects_positions.reshape(self.num_blocks, 3)
        self.objects_positions = objects_positions.copy()
        object_combinations = itertools.combinations(objects_positions, 2)
        object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

        self.goal_size = len(object_rel_distances)
        
        achieved_goal_binary = self._get_configuration(objects_positions).flatten()

        return {'observation': obs.copy(),
                'achieved_goal': objects_positions.flatten().copy(),
                'desired_goal': self.target_goal.copy(),
                'achieved_goal_binary': achieved_goal_binary.copy(),
                'desired_goal_binary': self.binary_goal.copy()}

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # print("sites offset: {}".format(sites_offset[0]))
        #for i in range(self.num_blocks):

            #site_id = self.sim.model.site_name2id('target{}'.format(i))
            #self.sim.model.site_pos[site_id] = self.goal[i*3:(i+1)*3] - sites_offset[i]

        self.sim.forward()

    def _grasp(self, obs, target_idx):
        success = False
        itr = 0
        observation = obs['observation']
        # Make sure to get high enough not to hit objects
        for _ in range(10):
            next_obs, r, d, info = self.step([0, 0, 1, 1])
            observation = next_obs['observation']
        # Reach object and grasp it
        while not success and itr < 20:
            action = np.concatenate((7 * (-observation[:3] + observation[10 + 15 * target_idx:13 + 15 * target_idx]), np.ones(1)))
            if np.linalg.norm(-observation[:3] + observation[10 + 15 * target_idx:13 + 15 * target_idx]) < 0.005:
                for _ in range(15):
                    action = [0., 0., 0.4, -1]
                    next_obs, r, d, info = self.step(action)
                    observation = next_obs['observation']
                    itr += 1
                success = True
            else:
                next_obs, r, d, info = self.step(action)
                observation = next_obs['observation']
                itr += 1
        obs['observation'] = observation
        return obs

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        self._sample_goal()

        self.sim.set_state(self.initial_state)
        
        # Guide learning by generating a stack in initialization
        if "above" in self.predicates or "right_close" in self.predicates:
            self.guide = True

        if self.guide:
            stack_level_proba = np.random.uniform()
            if stack_level_proba > 0.8:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
                z_stack = [0.525, 0.475, 0.425]
            else:
                stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
                z_stack = [0.475, 0.425]
            k = np.random.uniform()
            if k < 0.9:
                for i, obj_name in enumerate(self.object_names):
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                    assert object_qpos.shape == (7,)
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos

                    self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
            else:
                temp_rand = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                for i, obj_name in enumerate(self.object_names):
                    object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                    assert object_qpos.shape == (7,)
                    if i in stack:
                        object_qpos[2] = z_stack[stack.index(i)]
                        object_xpos = self.initial_gripper_xpos[:2] + temp_rand
                        object_qpos[:2] = object_xpos

                    else:
                        object_qpos[2] = 0.425
                        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                             self.obj_range,
                                                                                             size=2)
                        object_qpos[:2] = object_xpos

                    self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
                
        else:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        self.sim.forward()

        obs = self._get_obs()

        return obs

    def _generate_valid_goal(self):
        raise NotImplementedError

    def _sample_goal(self):
        # self.target_goal = self._generate_valid_goal()
        self.binary_goal = np.random.randint(2, size=9).astype(np.float32)
        self.target_goal = self.sample_continuous_goal_from_binary_goal(self.binary_goal.copy())
        return self.target_goal

    def set_goal(self, goal):
        self.target_goal = goal

    def _is_success(self, achieved_goal, desired_goal):
        return (achieved_goal == desired_goal).all()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(10):
            self.sim.step()

            # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def is_valid(self, binary_goal):
        possible_stacks = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
        actual_stacks = possible_stacks[np.where(np.array(binary_goal[-6:]) == 1.)]
        possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
        actual_close_pairs = possible_pairs[np.where(np.array(binary_goal[:3]) == 1.)]
        actual_close_pairs = [set(cp) for cp in actual_close_pairs]
        valid = True

        # check stacks are asymmetric
        for i, j in zip([3, 5, 7], [4, 6, 8]):
            if binary_goal[i] == 1 and binary_goal[j] == 1:
                valid = False
                break

        if actual_stacks.shape[0] == 1:
            remain = list(set([0, 1, 2]) - set(actual_stacks[0]))[0]
            stacked = actual_stacks[0][0]
            base = actual_stacks[0][1]

            # check that the stacked blocks are close
            if set([stacked, base]) not in actual_close_pairs:
                valid = False

            # check that if there is a stack, the third cube is either close to the base or close to both
            if set([stacked, remain]) in actual_close_pairs and set([base, remain]) not in actual_close_pairs:
                valid = False
        elif actual_stacks.shape[0] == 2:
            stacked1 = actual_stacks[0][0]
            base1 = actual_stacks[0][1]
            stacked2 = actual_stacks[1][0]
            base2 = actual_stacks[1][1]

            if base1 == base2:
                valid = False
            else:
                # check in stack that bottom and top are not close
                if stacked1 != stacked2:
                    if stacked1 == base2:
                        if set([stacked2, base1]) in actual_close_pairs:
                            valid = False
                    elif stacked2 == base1:
                        if set([stacked1, base2]) in actual_close_pairs:
                            valid = False
                    else:
                        raise ValueError
                else:
                    # check in pyramid that bases are close
                    if set([base1, base2]) not in actual_close_pairs:
                        valid = False
            # check that the stacked blocks are close
            if set([stacked1, base1]) not in actual_close_pairs:
                valid = False

            # check that the stacked blocks are close
            if set([stacked2, base2]) not in actual_close_pairs:
                valid = False


        elif actual_stacks.shape[0] > 2:
            # cannot have more than two stacks
            valid = False

        return  valid

    def sample_continuous_goal_from_binary_goal(self, binary_goal):

        valid_config = self.is_valid(binary_goal)



        if valid_config:
            possible_stacks = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
            actual_stacks = possible_stacks[np.where(np.array(binary_goal[-6:]) == 1.)]
            possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
            actual_close_pairs = possible_pairs[np.where(np.array(binary_goal[:3]) == 1.)]
            blocks_set = [False for _ in range(self.num_blocks)]
            positions = [None for _ in range(self.num_blocks)]
            if actual_stacks.shape[0] > 0:
                # There is at least one stack
                if actual_stacks.shape[0] == 1.:
                    over = False
                    counter_over = 0
                    while not over:
                        counter_over += 1
                        positions = [None for _ in range(self.num_blocks)]
                        over = True
                        # There exactly one stack
                        stack = actual_stacks[0]
                        z_stack = [0.475, 0.425]
                        offset_on_table = np.random.choice([-1, 1], size=2) * self.np_random.uniform(0.07, self.obj_range, size=2)
                        for i in stack:
                            object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
                            blocks_set[i] = True
                            positions[i] = np.array([object_xpos[0], object_xpos[1],  z_stack[np.where(stack == i)[0][0]]])

                        # Getting remaining blocks which were not placed
                        remain_block = [i for i, x in enumerate(blocks_set) if not x]
                        if len(remain_block) > 0:
                            if sum(binary_goal[:3]) == 1:
                                remain_block = remain_block[0]
                                stacked_block_id = actual_stacks[0][0]
                                base_block_id = actual_stacks[0][1]
                                # Added center to make sure the block stays on the table
                                ok = False
                                counter = 0
                                while not ok:
                                    counter += 1
                                    position = self.initial_gripper_xpos.copy()
                                    position[:2] += np.random.uniform(-self.obj_range, self.obj_range, 2)
                                    position[2] = 0.425
                                    cond1 =  np.linalg.norm(position - positions[base_block_id]) > self.predicate_threshold
                                    cond3 = position[0] < self.min_max_x[1] and position[0] > self.min_max_x[0]
                                    cond4 = position[1] < self.min_max_y[1] and position[1] > self.min_max_y[0]
                                    if cond1 and cond3 and cond4:
                                        ok = True
                                    if counter > 100:
                                        over = False
                                        break
                                positions[remain_block] = position.copy()
                            elif sum(binary_goal[:3]) == 2:
                                # remaining block is close to base but not close to stacked
                                remain_block = remain_block[0]
                                stacked_block_id = actual_stacks[0][0]
                                base_block_id = actual_stacks[0][1]
                                close_p = [set(cp) for cp in actual_close_pairs]
                                if set([stacked_block_id, remain_block]) in close_p and set([base_block_id, remain_block]) not in close_p:
                                    invalid_config = True
                                else:
                                    ok = False
                                    counter = 0
                                    while not ok:
                                        counter += 1
                                        position = positions[base_block_id].copy()
                                        position[:2] += np.random.uniform(-self.predicate_threshold, self.predicate_threshold, 2)
                                        position[2] = 0.425
                                        cond1 = np.linalg.norm(position - positions[stacked_block_id]) > self.predicate_threshold
                                        cond2 = np.linalg.norm(position - positions[base_block_id]) > np.sqrt(2) * 0.05
                                        cond3 = position[0] < self.min_max_x[1] and position[0] > self.min_max_x[0]
                                        cond4 = position[1] < self.min_max_y[1] and position[1] > self.min_max_y[0]
                                        cond5 = np.linalg.norm(position - positions[base_block_id]) < self.predicate_threshold
                                        if cond1 and cond2 and cond3 and cond4 and cond5:
                                            ok = True

                                        if counter > 100:
                                            over = False
                                            break
                                    positions[remain_block] = position.copy()
                            else:
                                # remaining block is close to both others
                                remain_block = remain_block[0]
                                stacked_block_id = actual_stacks[0][0]
                                base_block_id = actual_stacks[0][1]
                                # Added center to make sure the block stays on the table
                                ok = False
                                counter = 0
                                while not ok:
                                    counter += 1
                                    position = positions[base_block_id].copy()
                                    position[:2] += np.random.uniform(-self.predicate_threshold, self.predicate_threshold, 2)
                                    position[2] = 0.425
                                    cond1 = np.linalg.norm(position - positions[stacked_block_id]) < self.predicate_threshold
                                    cond2 = np.linalg.norm(position - positions[base_block_id]) > np.sqrt(2) * 0.05
                                    cond3 = position[0] < self.min_max_x[1] and position[0] > self.min_max_x[0]
                                    cond4 = position[1] < self.min_max_y[1] and position[1] > self.min_max_y[0]
                                    if cond1 and cond2 and cond3 and cond4:
                                        ok = True
                                    if counter > 100:
                                        over = False
                                        break

                                positions[remain_block] = position.copy()
                        if counter_over > 100:
                            valid_config = False
                            break
                elif actual_stacks.shape[0] > 1.:
                    # There are two stacks
                    if actual_stacks[0][0] == actual_stacks[1][0]:
                        # One block above two blocks (pyramid)
                        bot_block_1 = actual_stacks[0][1]
                        bot_block_2 = actual_stacks[1][1]
                        top_block = actual_stacks[0][0]
                        pos_on_table_1 = self.initial_gripper_xpos[:2].copy()
                        pos_on_table_1 += np.random.choice([-1, 1], size=2) * self.np_random.uniform(0.07, self.obj_range, size=2)
                        pos_on_table_2 = pos_on_table_1 + [0, 0.05]
                        pos_on_table_3 = pos_on_table_1 + [0, 0.025]
                        z = [0.425, 0.425, 0.475]
                        xy = [pos_on_table_1, pos_on_table_2, pos_on_table_3]
                        for i, block in enumerate([bot_block_1, bot_block_2, top_block]):
                            blocks_set[block] = True
                            positions[block] = np.array([xy[i][0], xy[i][1], z[i]])
                    else:
                        # A stack of 3 blocks
                        if actual_stacks[0][1] == actual_stacks[1][0]:
                            stack = [actual_stacks[0][0], actual_stacks[0][1], actual_stacks[1][1]]
                        else:
                            stack = [actual_stacks[1][0], actual_stacks[1][1], actual_stacks[0][1]]
                        z_stack = [0.525, 0.475, 0.425]
                        offset_on_table = np.random.choice([-1, 1], size=2) * self.np_random.uniform(0.07, self.obj_range, size=2)
                        for i in stack:
                            object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
                            blocks_set[i] = True
                            positions[i] = np.array([object_xpos[0], object_xpos[1], z_stack[stack.index(i)]])
                        if len(actual_close_pairs) < 3:
                            positions[stack[0]][0] += 0.02
                            positions[stack[0]][1] += 0.02
                else:
                    raise ValueError('should not happen')


            else:
                if sum(binary_goal[:3]) == 0:
                    # all far
                    over = False
                    counter_over = 0
                    while not over:
                        counter_over += 1
                        over = True
                        positions = []
                        for _ in range(3):
                            ok = False
                            counter = 0
                            while not ok:
                                counter += 1
                                pos = self.initial_gripper_xpos.copy()
                                pos[2] = 0.425
                                pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                                ok = True
                                for p in positions:
                                    if np.linalg.norm(p - pos) < self.predicate_threshold:
                                        ok = False
                                        break
                                if counter > 100:
                                    over = False
                                    break
                            positions.append(pos)
                        if counter_over > 100:
                            valid_config = False
                            break
                else:
                    possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
                    actual_close_pairs = possible_pairs[np.where(np.array(binary_goal[:3]) == 1.)]
                    if len(actual_close_pairs) == 3:
                        # all close
                        over = False
                        counter_over = 0
                        while not over:
                            counter_over += 1
                            over = True
                            positions = []
                            for _ in range(3):
                                ok = False
                                counter = 0
                                while not ok:
                                    counter += 1
                                    pos = self.initial_gripper_xpos.copy()
                                    pos[2] = 0.425
                                    pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                                    ok = True
                                    for p in positions:
                                        if np.linalg.norm(p - pos) < np.sqrt(2) * 0.05 or np.linalg.norm(p - pos) > self.predicate_threshold:
                                            ok = False
                                            break
                                    if counter > 100:
                                        over = False
                                        break
                                positions.append(pos)
                            if counter_over > 100:
                                valid_config = False
                                break
                    elif len(actual_close_pairs) == 2:
                        over = False
                        counter_over = 0
                        while not over:
                            counter_over += 1
                            over = True
                            pivot = list(set(actual_close_pairs[0]) & set(actual_close_pairs[1]))[0]
                            indexes = [i for i in itertools.chain(actual_close_pairs[0], actual_close_pairs[1]) if i != pivot]
                            indexes = [pivot] + indexes
                            positions = []
                            pivot = list(set(actual_close_pairs[0]) & set(actual_close_pairs[1]))[0]
                            pos = self.initial_gripper_xpos.copy()
                            pos[2] = 0.425
                            pos[:2] += self.np_random.uniform(-self.obj_range/2, self.obj_range/2, size=2)
                            positions.append(pos)
                            for _ in range(2):
                                ok = False
                                counter = 0
                                while not ok:
                                    counter += 1
                                    pos = self.initial_gripper_xpos.copy()
                                    pos[2] = 0.425
                                    pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                                    ok = False
                                    if len(positions) == 1:
                                        if np.linalg.norm(pos - positions[0]) > np.sqrt(2) * 0.05 and np.linalg.norm(pos - positions[0]) < self.predicate_threshold:
                                            ok = True
                                    if len(positions) == 2:
                                        cond1 = np.linalg.norm(pos - positions[0]) > np.sqrt(2) * 0.05 and np.linalg.norm(pos - positions[0]) < self.predicate_threshold
                                        cond2 =  np.linalg.norm(pos - positions[1]) > self.predicate_threshold
                                        if cond1 and cond2:
                                            ok = True
                                    if counter > 100:
                                        over = False
                                        break
                                positions.append(pos)
                            new_pos = [None] * 3
                            for i_order, i_ind in enumerate(indexes):
                                new_pos[i_ind] = positions[i_order]
                            positions = new_pos.copy()
                            if counter_over > 100:
                                valid_config = False
                                break
                    else:
                        over = False
                        counter_over = 1
                        while not over:
                            counter_over += 1
                            over = True
                            positions = []
                            pos = self.initial_gripper_xpos.copy()
                            pos[2] = 0.425
                            pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                            positions.append(pos)
                            for _ in range(2):
                                ok = False
                                counter = 0
                                while not ok:
                                    counter += 1
                                    pos = self.initial_gripper_xpos.copy()
                                    pos[2] = 0.425
                                    pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                                    ok = False
                                    if len(positions) == 1:
                                        if np.linalg.norm(pos - positions[0]) > np.sqrt(2) * 0.05 and np.linalg.norm(pos - positions[0]) < (self.predicate_threshold - 0.001):
                                            ok = True
                                    if len(positions) == 2:
                                        if np.linalg.norm(pos - positions[1]) > self.predicate_threshold and np.linalg.norm(pos - positions[0]) > self.predicate_threshold:
                                            ok = True

                                    if counter > 100:
                                        over = False
                                        break
                                positions.append(pos)
                            new_pos = [None] * 3
                            indexes = actual_close_pairs[0]
                            for i_order, i_ind in enumerate(indexes):
                                new_pos[i_ind] = positions[i_order]
                            for i_np, newp in enumerate(new_pos):
                                if newp is None:
                                    new_pos[i_np] = positions[-1]
                            positions = new_pos.copy()
                            if counter_over > 100:
                                valid_config = False
                                break


        if not valid_config:
            # print('invalid config')
            positions = []
            for _ in range(3):
                ok = False
                while not ok:
                    pos = self.initial_gripper_xpos.copy()
                    pos[2] = 0.425
                    pos[:2] += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    ok = True
                    for p in positions:
                        if np.linalg.norm(p - pos) < np.sqrt(2) * 0.05:
                            ok = False
                            break
                positions.append(pos)

        # for i, obj_name in enumerate(self.object_names):
        #     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
        #     object_qpos[:3] = positions[i]
        #     self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        return np.array(positions).flatten()


    def reset_goal(self, goal, init=None, biased_init=False):
        if init is not None:
            return self.reset_init(init, goal)

        self.binary_goal = goal.copy()
        self.target_goal = self.sample_continuous_goal_from_binary_goal(goal.copy())

        # If evaluation mode, generate blocks on the table with no stacks
        if not biased_init:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

            self.sim.forward()
            obs = self._get_obs()

            return obs

        p_stack_two = 0.7
        if np.random.uniform() > p_stack_two:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 3, replace=False))
            z_stack = [0.525, 0.475, 0.425]
        else:
            stack = list(np.random.choice([i for i in range(self.num_blocks)], 2, replace=False))
            z_stack = [0.475, 0.425]

        p_coplanar = 0.7
        idx_grasp = np.random.choice([i for i in range(self.num_blocks)])
        if np.random.uniform() < p_coplanar:
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                object_qpos[2] = 0.425
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                     self.obj_range,
                                                                                     size=2)
                object_qpos[:2] = object_xpos

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
        else:
            temp_rand = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            for i, obj_name in enumerate(self.object_names):
                object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
                assert object_qpos.shape == (7,)
                if i in stack:
                    object_qpos[2] = z_stack[stack.index(i)]
                    object_xpos = self.initial_gripper_xpos[:2] + temp_rand
                    object_qpos[:2] = object_xpos

                else:
                    object_qpos[2] = 0.425
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                    object_qpos[:2] = object_xpos

                    idx_grasp = i

                self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
            if len(stack) == self.num_blocks:
                idx_grasp = stack[0]

        self.sim.forward()
        obs = self._get_obs()

        if np.random.uniform() < 0.6:
            obs = self._grasp(obs, idx_grasp)
        return obs

    # def reset_init(self, init_config, target_goal):
    #     self.target_goal = target_goal
    #
    #     self.sim.set_state(self.initial_state)
    #     possible_stacks = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
    #     actual_stacks = possible_stacks[np.where(np.array(init_config[-6:]) == 1.)]
    #     blocks_set = [False for i in range(self.num_blocks)]
    #     positions = [None for i in range(self.num_blocks)]
    #     if actual_stacks.shape[0] > 0:
    #         # There is at least one stack
    #         if actual_stacks.shape[0] == 1.:
    #             # There exactly one stack
    #             stack = actual_stacks[0]
    #             z_stack = [0.475, 0.425]
    #             offset_on_table = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #             for i in stack:
    #                 object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
    #                 object_qpos[2] = z_stack[np.where(stack == i)[0][0]]
    #                 object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
    #                 object_qpos[:2] = object_xpos
    #                 blocks_set[i] = True
    #                 positions[i] = object_qpos[:3]
    #                 self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
    #         if actual_stacks.shape[0] > 1.:
    #             # There are two stacks
    #             if actual_stacks[0][0] == actual_stacks[1][0]:
    #                 # One block above two blocks (pyramid)
    #                 bot_block_1 = actual_stacks[0][1]
    #                 bot_block_2 = actual_stacks[1][1]
    #                 top_block = actual_stacks[0][0]
    #                 pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
    #                                                                                         size=2)
    #                 pos_on_table_2 = pos_on_table_1 + [0, 0.05]
    #                 pos_on_table_3 = pos_on_table_1 + [0, 0.025]
    #                 z = [0.425, 0.425, 0.475]
    #                 xy = [pos_on_table_1, pos_on_table_2, pos_on_table_3]
    #                 for i, block in enumerate([bot_block_1, bot_block_2, top_block]):
    #                     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[block]))
    #                     object_qpos[2] = z[i]
    #                     object_xpos = xy[i]
    #                     object_qpos[:2] = object_xpos
    #                     blocks_set[block] = True
    #                     positions[i] = object_qpos[:3]
    #                     self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[block]), object_qpos)
    #             else:
    #                 # A stack of 3 blocks
    #                 if actual_stacks[0][1] == actual_stacks[1][0]:
    #                     stack = [actual_stacks[0][0], actual_stacks[0][1], actual_stacks[1][1]]
    #                 else:
    #                     stack = [actual_stacks[1][0], actual_stacks[1][1], actual_stacks[0][1]]
    #                 z_stack = [0.525, 0.475, 0.425]
    #                 offset_on_table = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #                 for i in stack:
    #                     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
    #                     object_qpos[2] = z_stack[stack.index(i)]
    #                     object_xpos = self.initial_gripper_xpos[:2] + offset_on_table
    #                     object_qpos[:2] = object_xpos
    #                     blocks_set[i] = True
    #                     positions[i] = object_qpos[:3]
    #                     self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
    #         # Getting remaining blocks which were not placed
    #         remain_block = [i for i, x in enumerate(blocks_set) if not x]
    #         if len(remain_block) > 0:
    #             if sum(init_config[:3]) == 1:
    #                 remain_block = remain_block[0]
    #                 object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
    #                 object_qpos[2] = 0.425
    #                 # Added center to make sure the block stays on the table
    #                 center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
    #                 object_xpos = positions[np.where(blocks_set)[0][0]][:2] + \
    #                               np.array([-np.sign(center[0]) * 0.1, -np.sign(center[1]) * 0.1])
    #                 object_qpos[:2] = object_xpos
    #                 blocks_set[remain_block] = True
    #                 positions[remain_block] = object_qpos[:3]
    #                 self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)
    #             else:
    #                 # The remaining block is close to the stack
    #                 remain_block = remain_block[0]
    #                 object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
    #                 object_qpos[2] = 0.425
    #                 # Added center to make sure the block stays on the table
    #                 center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
    #                 object_xpos = positions[np.where(blocks_set)[0][0]][:2] + np.array([0, -np.sign(center[1]) * 0.05])
    #                 object_qpos[:2] = object_xpos
    #                 blocks_set[remain_block] = True
    #                 positions[remain_block] = object_qpos[:3]
    #                 self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)
    #
    #     else:
    #         if sum(init_config[:3]) == 0:
    #             # All blocks are far
    #             offset = self.np_random.uniform(0.1, self.obj_range)
    #             offset1 = self.np_random.uniform(0.1, self.obj_range)
    #             offsets = [[offset, -offset1], [-offset, offset1], [-offset1, -offset]]
    #             np.random.shuffle(offsets)
    #             for i, obj_name in enumerate(self.object_names):
    #                 object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
    #                 assert object_qpos.shape == (7,)
    #                 object_qpos[2] = 0.425
    #                 object_xpos = self.initial_gripper_xpos[:2] + offsets[i]
    #                 object_qpos[:2] = object_xpos
    #                 blocks_set[i] = True
    #                 positions[i] = object_qpos[:3]
    #                 self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
    #
    #         else:
    #             possible_pairs = np.array([[0, 1], [0, 2], [1, 2]])
    #             actual_close_pairs = possible_pairs[np.where(np.array(init_config[:3]) == 1.)]
    #             if len(actual_close_pairs) == 3:
    #                 # All blocks are close to each other
    #                 pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
    #                                                                                         size=2)
    #                 pos_on_table_2 = pos_on_table_1 + [0, 0.05]
    #                 pos_on_table_3 = pos_on_table_1 + [0.025, 0.05]
    #                 xy = [pos_on_table_1, pos_on_table_2, pos_on_table_3]
    #                 for i, obj_name in enumerate(self.object_names):
    #                     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
    #                     object_qpos[2] = 0.425
    #                     object_xpos = xy[i]
    #                     object_qpos[:2] = object_xpos
    #                     blocks_set[i] = True
    #                     positions[i] = object_qpos[:3]
    #                     self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)
    #             elif len(actual_close_pairs) == 2:
    #                 pivot = list(set(actual_close_pairs[0]) & set(actual_close_pairs[1]))[0]
    #                 indexes = [i for i in itertools.chain(actual_close_pairs[0], actual_close_pairs[1]) if i != pivot]
    #                 indexes = [pivot] + indexes
    #                 pos_on_tab_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,size=2)
    #                 pos_on_tab_2 = pos_on_tab_1 + np.array([0., 0.07])
    #                 pos_on_tab_3 = pos_on_tab_1 - np.array([0., 0.07])
    #                 xy = [pos_on_tab_1, pos_on_tab_2, pos_on_tab_3]
    #                 for i, pos in zip(indexes, xy):
    #                     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
    #                     object_qpos[2] = 0.425
    #                     object_xpos = pos
    #                     object_qpos[:2] = object_xpos
    #                     blocks_set[i] = True
    #                     positions[i] = object_qpos[:3]
    #                     self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
    #             else:
    #                 for pair in actual_close_pairs:
    #                     if not blocks_set[pair[0]]:
    #                         pos_on_table_1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
    #                                                                                                 size=2)
    #                     else:
    #                         pos_on_table_1 = positions[pair[0]][:2]
    #                     pos_on_table_2 = pos_on_table_1 + np.array([0, 0.07])
    #                     xy = [pos_on_table_1, pos_on_table_2]
    #                     for i, pos in zip(pair, xy):
    #                         object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[i]))
    #                         object_qpos[2] = 0.425
    #                         object_xpos = pos
    #                         object_qpos[:2] = object_xpos
    #                         blocks_set[i] = True
    #                         positions[i] = object_qpos[:3]
    #                         self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[i]), object_qpos)
    #                 if sum(blocks_set) != len(blocks_set):
    #                     remain_block = np.where(np.invert(blocks_set))[0][0]
    #                     object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(self.object_names[remain_block]))
    #                     object_qpos[2] = 0.425
    #                     # Added center to make sure the block stays on the table
    #                     center = positions[np.where(blocks_set)[0][0]][:2] - self.initial_gripper_xpos[:2]
    #                     object_xpos = positions[np.where(blocks_set)[0][0]][:2] + \
    #                                   np.array([-np.sign(center[0])*0.1, -np.sign(center[1])*0.1])
    #                     object_qpos[:2] = object_xpos
    #                     blocks_set[remain_block] = True
    #                     positions[remain_block] = object_qpos[:3]
    #                     self.sim.data.set_joint_qpos('{}:joint'.format(self.object_names[remain_block]), object_qpos)
    #
    #     self.sim.forward()
    #     obs = self._get_obs()
    #
    #     return obs
