import os ,sys

import logging
from settings_folder import settings
from game_config_handler_class import *
from game_handler_class import *
import file_handling
import msgs
#from common import utils
import json
import copy
import gym
from gym import spaces
from gym.utils import seeding
from algorithms.continuous.ddpg.OU import OU
import random
from gym_airsim.envs.airlearningclient import *
from utils import append_log_file
logger = logging.getLogger(__name__)



class AirSimEnv(gym.Env):
    #3.6.8 (self.airgym = None

    def __init__(self):

        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 256, 3))
        print('observation space shape', self.observation_space.shape)
        '''
        self.observation_space = spaces.Dict({"rgb": spaces.Box(low = 0, high=255, shape=(144, 256, 3)),
                                              "depth": spaces.Box(low = 0, high=255, shape=(144, 256,1)),
                                              "velocity": spaces.Box(low=-10, high=10, shape=(3,)),
                                              "position:":spaces.Box(low=np.Inf, high=np.NINF, shape=(4,))})
        '''


        self.action_space = spaces.Discrete(3) # right left and front

        self.airgym = AirLearningClient()
        #self.goal = self.airgym.circlepos()

        print("goal is",self.goal) # updated to a new goal


        self._seed()



    # This function was introduced (instead of the body to be merged into
    # __init__ because I need difficulty level as an argument but I can't
    # touch __init__ number of arguments

    def init_again(self, range_dic): #need this cause we can't pass arguments to
                                     # the main init function easily
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()
        self.sampleGameConfig()
        self.goal = self.airgym.circlepos()


    def getGoal(self): #there is no setting for the goal, cause you set goal
                   #indirectory by setting End
        return self.goal

    def state(self):
            return self.ss_state
            #return self.depth, self.velocity, self.position


    def print_msg_of_inspiration(self):
        if (self.success_count_within_window %2 == 0):
            print("---------------:) :) :) Success, Be Happy (: (: (:------------ !!!\n")
        elif (self.success_count_within_window %3 == 0):
            print("---------------:) :) :) Success, Shake Your Butt (: (: (:------------ !!!\n")
        else:
            print("---------------:) :) :) Success, Oh Yeah! (: (: (:------------ !!!\n")



    def _step(self, action):
        msgs.success = False
        msgs.meta_data = {}

        try:

            print("Action",action)
            self.stepN += 1
            self.total_step_count_for_experiment +=1

            collided = self.airgym.take_discrete_action(action)
            print("Collision:",collided)


            if distance < settings.success_distance_to_goal:
                self.success_count +=1
                done = True
                reward = 300
                self.print_msg_of_inspiration()
                self.success_count_within_window +=1
                self.success = True
                msgs.success = True

            elif self.stepN >= settings.nb_max_episodes_steps:
                done = True
                reward = 1-distance
                self.success = False

            elif collided == True:
                done = True
                reward = -20.0
                self.success = False
                #self.success = False
            else:
                done = False
                reward = 1-distance
                self.success = False



            #Todo: penalize for more crazy and unstable actions

            self.allLogs['distance'] = [float(distance)]
            self.distance_in_step.append(distance)
            self.count +=1
            self.position_in_step.append(str(now))
            info = {"x_pos":now[0], "y_pos":now[1]}
            # try:
            #     self.ss_state = self.airgym.get_SS_state(self)
            #     state = self.ss_state
            #     self.prev_state = state
            #
            # except Exception as e:
            #     done = False

            self.ss_state = self.airgym.get_SS_state(self)
            state = self.ss_state
            self.prev_state = state
            self.prev_info = info
            # if (done):
            #      self.on_episode_end()

            print("Reward",reward)
            return state, reward, done, info

        except Exception as e:
            print("------------------------- step failed ----------------  with"\
                    , e , " error")

            #self.game_handler.restart_game()
            self.airgym = AirLearningClient()
            return self.prev_state, 0, True, self.prev_info


    def _reset(self):
        try:
            if(settings.profile):
                if(self.stepN>1):
                    print("Avg loop rate:" + str(np.mean(self.loop_rate_list)))
                    self.all_loop_rates = copy.deepcopy(self.loop_rate_list)
                    print ("Action Time:" +str(np.mean(self.take_action_list)))
                    print("Collect State Time"+str(np.mean(self.clct_state_list)))
                if(self.stepN % 20 ==0):
                    print("Average Loop Rate:"+str(np.mean(self.all_loop_rates)))
            #self.random_pos = self.airgym.randomizing_agent()
            # print("enter reset")
            #
            # self.randomize_env()
            # print("done randomizing")
            # if(os.name=="nt"):
            #     connection_established = self.airgym.unreal_reset()
            #     if not connection_established:
            #         raise Exception
            # print("done unreal_resetting")

            time.sleep(2)
            self.airgym = AirLearningClient() #lets comment this to avoid pop up
            self.airgym.AirSim_reset()
            print("done arisim reseting")
            self.on_episode_start()
            print("done on episode start")
            #state = self.state()
            self.ss_state = self.airgym.get_SS_state(self)
            state = self.ss_state
            self.prev_state = state
            return state

        except Exception as e:
            print("------------------------- reset failed ----------------  with"\
                    , e , " error")
            #self.game_handler.restart_game()
            self.airgym = AirLearningClient()
