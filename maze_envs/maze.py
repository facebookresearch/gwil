### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import numpy as np
import tempfile
import gym
import xml.etree.ElementTree as ET
from .base import Env
import math

def construct_maze(maze_id=0, length=1):
    # define the maze to use
    if maze_id == 0:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        # structure = [
        #     [1, 1, 1, 1, 1],
        #     [1, 'r', 0, 0, 1],
        #     [1, 1, 1, 0, 1],
        #     [1, 1, 1, 0, 1],
        #     [1, 'g', 0, 0, 1],
        #     [1, 1, 1, 1, 1],
        # ]
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        # dense_reward_direction = [
        #     ["rr", "rr", "rr", "dd", "dd"],
        #     ["rr", "rr", "rr", "dd", "dd"],
        #     ["rr", "rr", "rr", "dd", "dd"],
        #     ["ll", "ll", "ll", "dd", "dd"],
        #     ["ll", "ll", "ll", "dl", "dl"],
        #     ["ll", "ll", "ll", "dl", "dl"],
        # ]

        dense_reward_direction = [
            [1, 1, 1, 1, 1],
            [1, 'r', 'r', 'd', 1],
            [1, 1, 1, 'd', 1],
            [1, 'l', 'l', 'l', 1],
            [1, 1, 1, 1, 1],
        ]
        # ##3:
        # dense_reward_direction = [
        #     ["r", "r", "r", "d", "d"],
        #     ["r", "r", "r", "d", "d"],
        #
        #     [1, 1, 1, "d", "d"],
        #     ["l", "l", "l", "l", "l"],
        #     ["l", "l", "l", "l", "l"],
        # ]

        # dense_reward = [
        #     [0, 1, 2, 3, 3],
        #     [0, 1, 2, 3, 3],
        #     [0, 1, 2, 4, 4],
        #     [8, 8, 7, 5, 5],
        #     [8, 8, 7, 6, 6],
        #     [8, 8, 7, 6, 6],
        # ]
        dense_reward = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 0, 4, 0],
            [0, 7, 6, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        return structure, dense_reward, dense_reward_direction
    elif maze_id == 1:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 'g', 1],
            [1, 1, 1, 1, 1]
        ]
        # dense_reward_direction = [
        #     ['rr', 'rr', 'rr', 'rr', 'rr'],
        #     ['rr', 'rr', 'rr', 'rr', 'rr'],
        #     ['rr', 'rr', 'rr', 'rr', 'rr']
        # ]
        dense_reward_direction = [
            [1, 1, 1, 1, 1],
            [1, 'r', 'r', 'r', 1],
            [1, 1, 1, 1, 1]
        ]
        dense_reward = [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 0, 0, 0]
        ]
        return structure, dense_reward, dense_reward_direction
    if maze_id == 2:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        dense_reward_direction = [
            [1, 1, 1, 1, 1],
            [1, 'l', 'l', 'l', 1],
            [1, 1, 1, 'u', 1],
            [1, 'r', 'r', 'u', 1],
            [1, 1, 1, 1, 1],
        ]
        dense_reward = [
            [0, 0, 0, 0, 0],
            [0, 7, 6, 5, 0],
            [0, 0, 0, 4, 0],
            [0, 1, 2, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        return structure, dense_reward, dense_reward_direction
    elif maze_id == 3:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'g', 0, 'r', 1],
            [1, 1, 1, 1, 1]
        ]
        dense_reward_direction = [
            [1, 1, 1, 1, 1],
            [1, 'l', 'l', 'l', 1],
            [1, 1, 1, 1, 1]
        ]
        dense_reward = [
            [0, 0, 0, 0, 0],
            [0, 3, 2, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        return structure, dense_reward, dense_reward_direction
    else:
        raise NotImplementedError("The provided MazeId is not recognized")

class Maze(Env):
    VISUALIZE = True
    SCALING = 8.0
    # MAZE_ID = 0
    DIST_REWARD = 0
    SPARSE_REWARD = 1000.0
    RANDOM_GOALS = False
    HEIGHT = 2

    # Fixed constants for agents
    SKILL_DIM = 2 # X, Y
    TASK_DIM = 4 # agent position, goal position.

    def __init__(self, model_path=None, maze_id=0):
        self.MAZE_ID = maze_id
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        
        # Initialize the maze and its parameters
        self.STRUCTURE, self.DENSE_REWARD, self.DENSE_REWARD_DIRECTION = construct_maze(maze_id=self.MAZE_ID, length=1)
        self.interm_goals = {}
        self.vel_rew = {}

        torso_x, torso_y = self.get_agent_start()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(self.DENSE_REWARD)):
            for j in range(len(self.DENSE_REWARD[0])):
                minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
                maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
                miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
                maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
                self.interm_goals[(minx, maxx, miny, maxy)] = self.DENSE_REWARD[i][j]
                self.vel_rew[(minx, maxx, miny, maxy)] = self.DENSE_REWARD_DIRECTION[i][j]

        tree = ET.parse(model_path)
        worldbody = tree.find(".//worldbody") 
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    self.current_goal_pos = (i,j)
                if (isinstance(self.STRUCTURE[i][j], int) or isinstance(self.STRUCTURE[i][j], float)) \
                    and self.STRUCTURE[i][j] > 0:
                    height = float(self.STRUCTURE[i][j])
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.SCALING - torso_x,
                                        i * self.SCALING - torso_y,
                                        self.HEIGHT / 2 * height),
                        size="%f %f %f" % (0.5 * self.SCALING,
                                        0.5 * self.SCALING,
                                        self.HEIGHT / 2 * height),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="%f %f 0.3 1" % (height * 0.3, height * 0.3)
                    )

        if self.VISUALIZE:
            world_body = tree.find(".//worldbody")
            waypoint_elem = ET.Element('body')
            waypoint_elem.set("name", "waypoint")
            waypoint_elem.set("pos", "0 0 " + str(self.SCALING/10))
            waypoint_geom = ET.SubElement(waypoint_elem, "geom")
            waypoint_geom.set("conaffinity", "0")
            waypoint_geom.set("contype", "0")
            waypoint_geom.set("name", "waypoint")
            waypoint_geom.set("pos", "0 0 0")
            waypoint_geom.set("rgba", "0.2 0.9 0.2 0.8")
            waypoint_geom.set("size", str(self.SCALING/10))
            waypoint_geom.set("type", "sphere")
            world_body.insert(-1, waypoint_elem)
            xml_path = model_path

        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_path)
        
        # Get the list of possible segments of the maze to be the goal.
        self.possible_goal_positions = list()
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 0 or self.STRUCTURE[i][j] == 'g':
                    self.possible_goal_positions.append((i,j))
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

        super(Maze, self).__init__(model_path=xml_path)

    def sample_goal_pos(self):
        if not self.RANDOM_GOALS:
            return
        cur_x, cur_y = self.current_goal_pos
        self.STRUCTURE[cur_x][cur_y] = 0
        new_x, new_y = self.possible_goal_positions[self.np_random.randint(low=0, high=len(self.possible_goal_positions))]
        self.STRUCTURE[new_x][new_y] = 'g'
        self.current_goal_pos = (new_x, new_y)
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

    def get_agent_start(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'r':
                    return j * self.SCALING, i * self.SCALING
        assert False

    def get_goal_range(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
                    maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
                    miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
                    maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy


    def xy_to_discrete(self, x, y):
        x = x + self._init_torso_x + self.SCALING * 0.5
        y = y + self._init_torso_y + self.SCALING * 0.5
        i = int(x//self.SCALING)
        j = int(y//self.SCALING)
        return i,j



    # def get_dense_reward2(self):
    #     # we need to remove int_torso to get mujoco positions
    #     x, y = self.get_body_com("torso")[:2]
    #     i, j = self.xy_to_discrete(x,y)
    #     # i = max(i,0)
    #     # j = max(i,0)
    #     # i = min(i, len(self.DENSE_REWARD_DIRECTION)-1)
    #     # j = min(j, len(self.DENSE_REWARD_DIRECTION[0])-1)
    #     # return self.DENSE_REWARD[i][j]*10
    #     try:
    #         direction = self.DENSE_REWARD_DIRECTION[i][j]
    #     except:
    #         print("wrong coord should not happen")
    #         return -100
    #     if direction=='i':
    #
    #         return -100
    #     #     miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
    #     #     maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
    #     #     center = (miny+maxy)/2
    #     #     if y + self._init_torso_y + self.SCALING * 0.5>center:
    #     #         i = i+1
    #     #     else:
    #     #         i = i-1
    #     #     direction = self.DENSE_REWARD_DIRECTION[i][j]
    #     if direction=='j':
    #         return -100
    #     #     minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
    #     #     maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
    #     #     center = (minx+maxx)/2
    #     #     if x + self._init_torso_x + self.SCALING * 0.5>center:
    #     #         j=j+1
    #     #     else:
    #     #         j=j-1
    #     #     direction = self.DENSE_REWARD_DIRECTION[i][j]
    #     base_reward = self.DENSE_REWARD[i][j]
    #     r_pos = 0
    #     r_vel = 0
    #     for dir in direction:
    #         if dir=='r':
    #             r_vel+= self.sim.data.qvel.flat[:][0]
    #             minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
    #             maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
    #             perturb = (x-minx)/(maxx-minx)
    #             perturb = max(perturb,0)
    #             perturb = min(perturb,1)
    #             r_pos += (base_reward+(perturb))*10
    #         elif dir == 'u':
    #             r_vel-=self.sim.data.qvel.flat[:][1]
    #             miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
    #             maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
    #             perturb = (maxy-y)/(maxy-miny)
    #             perturb = max(perturb, 0)
    #             perturb = min(perturb, 1)
    #             r_pos += (base_reward+(perturb))*10
    #         elif dir == 'l':
    #             r_vel -= self.sim.data.qvel.flat[:][0]
    #             minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
    #             maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
    #             perturb = (maxx - x) / (maxx - minx)
    #             perturb = max(perturb, 0)
    #             perturb = min(perturb, 1)
    #             r_pos += (base_reward+(perturb))*10
    #         elif dir == 'd':
    #             r_vel+= self.sim.data.qvel.flat[:][1]
    #             miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
    #             maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
    #             perturb = (y - miny) / (maxy - miny)
    #             perturb = max(perturb, 0)
    #             perturb = min(perturb, 1)
    #             r_pos += (base_reward+(perturb))*10
    #     return r_pos

    def _get_obs(self):
        return NotImplemented

    def step(self, action):
        # import pdb; pdb.set_trace()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        # Compute the reward
        minx, maxx, miny, maxy = self.goal_range
        x, y = self.get_body_com("torso")[:2]
        reward = 0
        if minx <= x <= maxx and miny <= y <= maxy:
            reward += self.SPARSE_REWARD
            done = True
        else:
            done = False
        # if self.DIST_REWARD > 0:
        #     # adds L2 reward
        #     reward += -self.DIST_REWARD * np.linalg.norm(self.skill_obs(obs)[:2] - self.center_goal)

        dense_reward = self.get_dense_reward()+reward-1

        obs_dic = self.get_obs_dic()
        
        return obs, reward, done, {'is_success' : done, 'dense_reward':dense_reward, 'dmc_obs': obs_dic}

    def reset(self):
        return NotImplemented

class MazeEnd_PointMass(Maze):
    ASSET = 'point_mass.xml'
    AGENT_DIM = 2
    FRAME_SKIP = 3

    def __init__(self, maze_id=0):
        super(MazeEnd_PointMass, self).__init__(maze_id=maze_id)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:],
            self.get_body_com("torso")[:2],
            self.center_goal,
        ])

    def get_obs_dic(self):
        # import pdb; pdb.set_trace()
        return {
            'velocity': self.sim.data.qvel.flat[:],
            'position': self.get_body_com("torso")[:2],
            'position_torso': self.get_body_com("torso")[:2]
        }

    def get_dense_reward(self):
        # we need to remove int_torso to get mujoco positions
        rew_pos = 0
        rew_vel = 0
        x, y = self.get_body_com("torso")[:2]
        for key in self.interm_goals:
            minx, maxx, miny, maxy = key
            if minx <= x <= maxx and miny <= y <= maxy:
                rew_pos+=self.interm_goals[key]
                if self.vel_rew[key]=='r':
                    rew_vel +=self.sim.data.qvel.flat[:][0]
                elif self.vel_rew[key] == 'u':
                    rew_vel -=self.sim.data.qvel.flat[:][1]
                elif self.vel_rew[key] == 'l':
                    rew_vel -= self.sim.data.qvel.flat[:][0]
                elif self.vel_rew[key] == 'd':
                    rew_vel += self.sim.data.qvel.flat[:][1]
                break
        # print(f"rew_pos: {rew_pos}, rew_vel: {rew_vel}")
        return rew_pos+rew_vel
        # i, j = self.xy_to_discrete(x,y)
        # # i = max(i,0)
        # # j = max(i,0)
        # # i = min(i, len(self.DENSE_REWARD_DIRECTION)-1)
        # # j = min(j, len(self.DENSE_REWARD_DIRECTION[0])-1)
        # # return self.DENSE_REWARD[i][j]*10
        # try:
        #     direction = self.DENSE_REWARD_DIRECTION[i][j]
        # except:
        #     # print("wrong coord should not happen")
        #     return -100
        # if direction==1:
        #
        #     return -100
        # #     miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
        # #     maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
        # #     center = (miny+maxy)/2
        # #     if y + self._init_torso_y + self.SCALING * 0.5>center:
        # #         i = i+1
        # #     else:
        # #         i = i-1
        # #     direction = self.DENSE_REWARD_DIRECTION[i][j]
        # # if direction=='j':
        # #     return -100
        # #     minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
        # #     maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
        # #     center = (minx+maxx)/2
        # #     if x + self._init_torso_x + self.SCALING * 0.5>center:
        # #         j=j+1
        # #     else:
        # #         j=j-1
        # #     direction = self.DENSE_REWARD_DIRECTION[i][j]
        # base_reward = self.DENSE_REWARD[i][j]
        # r_pos = 0
        # r_vel = 0
        # for dir in direction:
        #     if dir=='r':
        #         r_vel+= self.sim.data.qvel.flat[:][0]
        #         minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
        #         maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
        #         perturb = (x-minx)/(maxx-minx)
        #         perturb = max(perturb,0)
        #         perturb = min(perturb,1)
        #         r_pos += (base_reward+(perturb))*10
        #     elif dir == 'u':
        #         r_vel-=self.sim.data.qvel.flat[:][1]
        #         miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
        #         maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
        #         perturb = (maxy-y)/(maxy-miny)
        #         perturb = max(perturb, 0)
        #         perturb = min(perturb, 1)
        #         r_pos += (base_reward+(perturb))*10
        #     elif dir == 'l':
        #         r_vel -= self.sim.data.qvel.flat[:][0]
        #         minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
        #         maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
        #         perturb = (maxx - x) / (maxx - minx)
        #         perturb = max(perturb, 0)
        #         perturb = min(perturb, 1)
        #         r_pos += (base_reward+(perturb))*10
        #     elif dir == 'd':
        #         r_vel+= self.sim.data.qvel.flat[:][1]
        #         miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
        #         maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
        #         perturb = (y - miny) / (maxy - miny)
        #         perturb = max(perturb, 0)
        #         perturb = min(perturb, 1)
        #         r_pos += (base_reward+(perturb))*10
        # return r_pos
    
    def reset(self):
        self.sim.reset()
        self.sample_goal_pos()
        if self.VISUALIZE:
            self.model.body_pos[-2][:2] = self.center_goal
        qpos = self.init_qpos + self.np_random.uniform(low=-self.SCALING/10.0, high=self.SCALING/10.0, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
