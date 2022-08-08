

import airsim # sudo pip install airsim

import numpy as np
import math
import time
import cv2
import settings
import os
import gym
import random
import tempfile
import matplotlib.pyplot as plt
from .random import *
from PIL import Image
from pylab import array, uint8, arange
from random import seed
from random import random
from random import randint
#from random import RandGen
import numpy as np

import msgs

COLORS = {
    'red'   : np.array([255, 0, 0]),
    'blue'  : np.array([0, 0, 255]),
    'orange' : np.array([255,165,0])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'blue'  : 1,
    'orange' : 2
}



class AirLearningClient(airsim.MultirotorClient):
    def __init__(self):



        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient(settings.ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        #self.client.moveToPositionAsync(-5, 11, -2, 5).join()

        #self.client.hoverAsync().join()



        print("---------------------------------------------------------")

        #seed(50) #comment

        pose = self.client.simGetVehiclePose()



        # teleport the drone to any position
        # pose.position.x_val = randint(-1, 1) #gives an random interger to the agent between the given range
        pose.position.x_val = np.random.uniform(-2.5,2.5) #uncomment for randomizing the agent
        #pose.position.x_val = -20.13
        #pose.position.y_val = -12 #uncomment if you want to give manual values to the agent otherwise use the above thing
        pose.position.y_val = np.random.uniform(-2.5,2.5) #can be 24 but 23 ##uncomment for randomizing the agent


        self.client.simSetVehiclePose(pose, True, "Drone1")

        #time.sleep(5)



        self.home_pos = self.client.simSetVehiclePose(pose, True, "Drone1")

        print("Randomize coordiantes",pose.position.x_val,pose.position.y_val,-2)
        #
        x = pose.position.x_val #uncomment for randomizing the agent
        y = pose.position.y_val #uncomment for randomizing the agent

        self.home_ori = self.client.getOrientation()


        # x=random.uniform(0,35)
        # y=random.uniform(0,35)
        self.z = -2 #1 #4 1 if the circle is too low #0.4 for realworld
        print("z",self.z)






        self.folder = 'images'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        cam = '0'

        self.requests = [airsim.ImageRequest(cam, airsim.ImageType.Scene,)]
                                          #False,True)]
        self.i = 0

        # Image responses
        self.responses = []
        self.seed()

    def land(self):
        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed:
            print("already landed...")
        else:
            print("landing...")
        self.client.landAsync().join()

    #next 3 functions for getting the position

    '''--------------------------------------------------------------
    Function:  circlepos, dronepos
    Dev:  Tejaswini
    Date:    9.01.2022
    Description: Get the Orientation and position of the circle to calculate x, y, z values
    --------------------------------------------------------------'''
    def redcirclepos(self):

        circlepos = self.client.simGetObjectPose("wweed_3").position
        #print(circlepos)ch
        x = circlepos.x_val
        y = circlepos.y_val
        z = circlepos.z_val
        #circlepos.position.x_val = np.random.uniform(-2.5,2.5)
        #print("redpos",np.array([x, y, z]))
        return np.array([x, y, z])

    def bluecirclepos(self):

        bluecirclepos = self.client.simGetObjectPose("wweed4").position
        #print(circlepos)ch
        x = bluecirclepos.x_val
        y = bluecirclepos.y_val
        z = bluecirclepos.z_val
        #circlepos.position.x_val = np.random.uniform(-2.5,2.5)
        #print("bluepos",np.array([x, y, z]))
        return np.array([x, y, z])

    def cubepos(self):

        cubepos = self.client.simGetObjectPose("1M_Cube_5").position
        #print(circlepos)
        x = cubepos.x_val
        y = cubepos.y_val
        z = cubepos.z_val
        print(np.array([x, y, z]))
        return np.array([x, y, z])
    def dronepos(self):

         drone_pos= self.client.simGetVehiclePose(vehicle_name = "Drone1").position
         x = drone_pos.x_val
         y = drone_pos.y_val
         z = drone_pos.z_val
         print(np.array([x, y, z]))
         return np.array([x, y, z])

    def goal_direction(self, goal, pos):

        pitch, roll, yaw = self.client.getPitchRollYaw()
        yaw = math.degrees(yaw)

        pos_angle = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)

        return ((math.degrees(track) - 180) % 360) - 180

    def get_imudata(self):

        imu_data = self.client.getImuData(imu_name = "Imu", vehicle_name = "Drone1")


        return imu_data

    def getConcatState(self, goal):

        now = self.drone_pos()
        track = self.goal_direction(goal, now)
        encoded_depth = self.getScreenDepthVis(track)
        encoded_depth_shape = encoded_depth.shape
        encoded_depth_1d = encoded_depth.reshape(1, encoded_depth_shape[0]*encoded_depth_shape[1])

        #ToDo: Add RGB, velocity etc
        if(settings.position):
            pos = self.get_distance(goal)
            concat_state = np.concatenate((encoded_depth_1d, pos), axis = None)
            concat_state_shape = concat_state.shape
            concat_state = concat_state.reshape(1, concat_state_shape[0])
            concat_state = np.expand_dims(concat_state, axis=0)
        else:
            concat_state = encoded_depth_1d

        return concat_state

    def getScreenGrey(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if((responses[0].width !=0 or responses[0].height!=0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            #print(grey.shape)
        else:
            print("Something bad happened! Resetting AirSim!")
            self.AirSim_reset()
            grey = np.ones(144,256)
        return grey



    def getScreenRGB(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()
            rgb = np.ones(144, 256, 3)
        return rgb

    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        #responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis,True, False)])

        if(responses == None):
            print("Camera is not returning image!")
            print("Image size:"+str(responses[0].height)+","+str(responses[0].width))
        else:
            img1d = np.array(responses[0].image_data_float, dtype=np.float)

        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        if((responses[0].width!=0 or responses[0].height!=0)):
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        else:
            print("Something bad happened! Resetting AirSim!")
            self.AirSim_reset()
            img2d = np.ones((144, 256))

        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))

        factor = 10
        maxIntensity = 255.0  # depends on dtype of image data

        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark
        newImage1 = (maxIntensity) * (image / maxIntensity) ** factor
        newImage1 = array(newImage1, dtype=uint8)

        # small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
        small = cv2.resize(newImage1, (0, 0), fx=1.0, fy=1.0)

        cut = small[20:40, :]
        # print(cut.shape)

        info_section = np.zeros((10, small.shape[1]), dtype=np.uint8) + 255
        info_section[9, :] = 0

        line = np.int((((track - -180) * (small.shape[1] - 0)) / (180 - -180)) + 0)
        '''
        print("\n")
        print("Track:"+str(track))
        print("\n")
        print("(Track - -180):"+str(track - -180))
        print("\n")
        print("Num:"+str((track - -180)*(100-0)))
        print("Num_2:"+str((track+180)*(100-0)))
        print("\n")
        print("Den:"+str(180 - -180))
        print("Den_2:"+str(180+180))
        print("line:"+str(line))
        '''
        if line != (0 or small.shape[1]):
            info_section[:, line - 1:line + 2] = 0
        elif line == 0:
            info_section[:, 0:3] = 0
        elif line == small.shape[1]:
            info_section[:, info_section.shape[1] - 3:info_section.shape[1]] = 0

        total = np.concatenate((info_section, small), axis=0)
        #cv2.imwrite("test.png",total)
        # cv2.imshow("Test", total)
        # cv2.waitKey(0)
        #total = np.reshape(total, (154,256))
        return total

    def Gauss(self,x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    def get_sensor_read(self,dist):
        pars = np.array([398.96920648,  -2.58025025,   5.08711473])

        max_read = self.Gauss(0,*pars)
        read = self.Gauss(dist*3.28084,*pars)
        noise = np.random.normal(0,2)
        return((read+noise)/(max_read))

    def seed(self, seed=None):
        self.rand = RandGen(seed)
        return [seed]

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high]
        """

        return self.rand.int(low, high)





    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def target_pos(self):
        boxPos = []

        boxPos.append(self.redcirclepos())
        #print("redboxPos",boxPos)
        boxPos.append(self.bluecirclepos()) # bluecirclepos() it was for blue sphere #cubepos for box
        #print("boxPos",boxPos)
        boxColors = ['red', 'blue'] #orange for box

        boxIdx = self._rand_int(0, len(boxPos)) #0
        print("Boxid",boxIdx) #suppose 0
        self.target_pos = boxPos[boxIdx] #position of red
        print("target pos",self .target_pos) #position of red circle
        self.target_color = boxColors[boxIdx] #color of red
        print("target_color",self.target_color)

        self.target_Idx = boxIdx
        print("test", self.target_color, self.target_pos)
        return self.target_color, self.target_pos




    def get_SS_state(self,state):

        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if((responses[0].width !=0 or responses[0].height!=0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            grey = rgb

            #grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            dim = (80,60)
            grey = cv2.resize(grey, dim, interpolation = cv2.INTER_AREA)

            '''
            from PIL import Images

            image = Image.fromarray(grey)
            grey = np.array(image.resize((244,324)).convert("L"))
            '''
            # cv2.imshow("Greyscale",grey)
            # cv2.waitKey(1)
            grey = grey.reshape([80,60,3])


        else:
            print("Something bad happened! Resetting AirSim!")
            self.AirSim_reset()
            grey = np.ones(80,60)


        self.mission = 'go to the %s sphere' % self.target_color

        output = {"image": grey, "mission":self.mission}

        print("mission",self.mission)
        print('go to {} of sphere'.format(self.target_color))
        print("output",output)


        return self.target_pos, output
        '''


        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if((responses[0].width !=0 or responses[0].height!=0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)
            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            from PIL import Image

            image = Image.fromarray(grey)
            grey = np.array(image.resize((244,324)).convert("L"))

        else:
            print("Something bad happened! Resetting AirSim!")
            self.AirSim_reset()
            grey = np.ones(244,324)
            # cv2.imshow("Greyscale",grey)
            # cv2.waitKey(1)
        return grey.reshape([244,324,1])

        '''






        # self.drone_state = self.client.getMultirotorState()
        #
        # self.state["prev_position"] = self.client.state["position"]
        # self.state["position"] = self.client.drone_state.kinematics_estimated.position
        # self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        #return rgb

    def new(self):
        self.responses.append(self.client.simGetImages(self.requests) )
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if((responses[0].width !=0 or responses[0].height!=0)):
            img_rgba = img1d.reshape(response.height, response.width, 4)

            rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            dim = (80,60)
            rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
            #rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            #grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            from PIL import Image

            #image = Image.fromarray(grey)
            #grey = np.array(image.resize((324, 244)).convert("L"))

        filename = os.path.join(self.folder,'{:05d}.png'.format(self.i))
            # The image data
        cv2.imwrite(filename,rgb)
        self.i += 1
            # data = self.responses[i][0].image_data_uint8
            # # Write image data
            # with open(filename,'wb') as FILE:
            #     # Write the bytes to file
            #     FILE.write(grey)

        print("Saving images to %s" %filename)


    def drone_pos(self):
        x = self.client.getPosition().x_val
        y = self.client.getPosition().y_val
        z = self.client.getPosition().z_val

        return np.array([x, y, z])

    def drone_velocity(self):
        v_x = self.client.getVelocity().x_val
        v_y = self.client.getVelocity().y_val
        v_z = self.client.getVelocity().z_val

        return np.array([v_x, v_y, v_z])

    def get_distance(self, goal):
        now = self.client.getPosition()
        xdistance = (goal[0] - now.x_val)
        ydistance = (goal[1] - now.y_val)
        euclidean = np.sqrt(np.power(xdistance,2) + np.power(ydistance,2))

        return np.array([xdistance, ydistance, euclidean])

    def get_circle_distance(self):

        pass

    def get_velocity(self):
        return np.array([self.client.get_velocity().x_val, self.client.get_velocity().y_val, self.client.get_velocity().z_val])

    def AirSim_reset(self):

        self.client.reset()
        time.sleep(0.2)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.2)
        #self.goal = self.target_pos() #comment
        self.client.moveByVelocityAsync(0, 0, -self.z, 2, drivetrain=0, vehicle_name='').join() #change to 4 when height 4
        self.client.moveByVelocityAsync(0, 0, 0, 1, drivetrain=0, vehicle_name='').join()


    def unreal_reset(self):
        self.client, connection_established = self.client.resetUnreal()
        return connection_established

    def take_continious_action(self, action):

        if(msgs.algo == 'DDPG'):
            pitch = action[0]
            roll = action[1]
            throttle = action[2]
            yaw_rate = action[3]
            duration = action[4]
            self.client.moveByAngleThrottleAsync(pitch, roll, throttle, yaw_rate, duration, vehicle_name='').join()
        else:
            #pitch = np.clip(action[0], -0.261, 0.261)
            #roll = np.clip(action[1], -0.261, 0.261)
            #yaw_rate = np.clip(action[2], -3.14, 3.14)
            if(settings.move_by_velocity):
                vx = np.clip(action[0], -1.0, 1.0)
                vy = np.clip(action[1], -1.0, 1.0)
                #print("Vx, Vy--------------->"+str(vx)+", "+ str(vy))
                #self.client.moveByAngleZAsync(float(pitch), float(roll), -6, float(yaw_rate), settings.duration_ppo).join()
                self.client.moveByVelocityZAsync(float(vx), float(vy), -6, 0.5, 1, yaw_mode=airsim.YawMode(True, 0)).join()
            elif(settings.move_by_position):
                pos = self.drone_pos()
                delta_x = np.clip(action[0], -1, 1)
                delta_y = np.clip(action[1], -1, 1)
                self.client.moveToPositionAsync(float(delta_x + pos[0]), float(delta_y + pos[1]), -6, 0.9, yaw_mode=airsim.YawMode(False, 0)).join()
        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)
        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)

        return collided

    def TotalRewards(self):

        pass


        #Todo : Stabilize drone
        #self.client.moveByAngleThrottleAsync(0, 0,1,0,2).join()

        #TODO: Get the collision info and use that to reset the simuation.
        #TODO: Put some sleep in between the calls so as not to crash on the same lines as DQN
    def straight(self, speed, duration):
        pitch, roll, yaw  = self.client.getPitchRollYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 1).join()
        print("straight, rate:" + str(speed) + " duration:"+ str(duration))
        start = time.time()
        return start, duration

    def backup(self, speed, duration):
        pitch, roll, yaw = self.client.getPitchRollYaw()
        vx = math.cos(yaw) * speed * -0.5
        vy = math.sin(yaw) * speed * -0.5
        self.client.moveByVelocityZAsync(-vx, -vy, self.z, duration, 0).join()
        start = time.time()
        return start, duration

    def yaw_right(self, rate, duration):
        self.client.rotateByYawRateAsync(rate, duration).join()
        print("yaw_right, rate:" + str(rate) + " duration:"+ str(duration))
        start = time.time()
        return start, duration

    def yaw_left(self, rate, duration):
        print("yaw_left, rate:" + str(rate) + " duration:"+ str(duration))
        self.client.rotateByYawRateAsync(-rate, duration).join()
        start = time.time()
        return start, duration

    def pitch_up(self, duration):
        self.client.moveByVelocityZAsync(0.5,0,self.z,duration,0).join()
        start = time.time()
        return start, duration

    def pitch_down(self, duration):
        self.client.moveByVelocityZAsync(0.5,0.5,self.z,duration,0).join()
        start = time.time()
        return start, duration

    def move_forward_Speed(self, speed_x = 0.5, speed_y = 0.5, duration = 0.5):
        pitch, roll, yaw  = self.client.getPitchRollYaw()
        vel = self.client.getVelocity()
        vx = math.cos(yaw) * speed_x - math.sin(yaw) * speed_y
        vy = math.sin(yaw) * speed_x + math.cos(yaw) * speed_y
        if speed_x <= 0.01:
            drivetrain = 0
            #yaw_mode = YawMode(is_rate= True, yaw_or_rate = 0)
        elif speed_x > 0.01:
            drivetrain = 0
            #yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)
        self.client.moveByVelocityZAsync(vx = (vx +vel.x_val)/2 ,
                             vy = (vy +vel.y_val)/2 , #do this to try and smooth the movement
                             z = self.z,
                             duration = duration,
                             drivetrain = drivetrain,
                            ).join()
        start = time.time()
        return start, duration

    def take_discrete_action(self, action):

       # check if copter is on level cause sometimes he goes up without a reason
        """
        x = 0
        while self.client.getPosition().z_val < -7.0:
            self.client.moveToZAsync(-6, 3).join()
            time.sleep(1)
            print(self.client.getPosition().z_val, "and", x)
            x = x + 1
            if x > 10:
                return True

        start = time.time()
        duration = 0
        """

        # if action == 0:
        #     start, duration = self.straight(settings.mv_fw_spd_5, settings.mv_fw_dur)
        # if action == 1:
        #     start, duration = self.straight(settings.mv_fw_spd_4, settings.mv_fw_dur)
        # if action == 2:
        #     start, duration = self.straight(settings.mv_fw_spd_3, settings.mv_fw_dur)
        # if action == 3:
        #     start, duration = self.straight(settings.mv_fw_spd_2, settings.mv_fw_dur)
        # if action == 4:
        #     start, duration = self.straight(settings.mv_fw_spd_1, settings.mv_fw_dur)
        # if action == 5:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_5, settings.mv_fw_spd_5, settings.mv_fw_dur)
        # if action == 6:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_4, settings.mv_fw_spd_4, settings.mv_fw_dur)
        # if action == 7:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, settings.mv_fw_spd_3, settings.mv_fw_dur)
        # if action == 8:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, settings.mv_fw_spd_2, settings.mv_fw_dur)
        # if action == 9:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, settings.mv_fw_spd_1, settings.mv_fw_dur)
        # if action == 10:
        #     start, duration = self.backup(settings.mv_fw_spd_5, settings.mv_fw_dur)
        # if action == 11:
        #     start, duration = self.backup(settings.mv_fw_spd_4, settings.mv_fw_dur)
        # if action == 12:
        #     start, duration = self.backup(settings.mv_fw_spd_3, settings.mv_fw_dur)
        # if action == 13:
        #     start, duration = self.backup(settings.mv_fw_spd_2, settings.mv_fw_dur)
        # if action == 14:
        #     start, duration = self.backup(settings.mv_fw_spd_1, settings.mv_fw_dur)
        # if action == 15:
        #     start, duration = self.yaw_right(settings.yaw_rate_1_1, settings.rot_dur)
        # if action == 16:
        #     start, duration = self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur)
        # if action == 17:
        #     start, duration = self.yaw_right(settings.yaw_rate_1_4, settings.rot_dur)
        # if action == 18:
        #     start, duration = self.yaw_right(settings.yaw_rate_1_8, settings.rot_dur)
        # if action == 19:
        #     start, duration = self.yaw_right(settings.yaw_rate_1_16, settings.rot_dur)
        # if action == 20:
        #     start, duration = self.yaw_right(settings.yaw_rate_2_1, settings.rot_dur)
        # if action == 21:
        #     start, duration = self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        # if action == 22:
        #     start, duration = self.yaw_right(settings.yaw_rate_2_4, settings.rot_dur)
        # if action == 23:
        #     start, duration = self.yaw_right(settings.yaw_rate_2_8, settings.rot_dur)
        # if action == 24:
        #     start, duration = self.yaw_right(settings.yaw_rate_2_16, settings.rot_dur)
        if(settings.velocity_noise==False):
            delta=0
        else:
            delta = np.random.normal(0,settings.noise_std)

        #print(delta)

        # if action ==0:
        #     start, duration = self.move_forward_Speed(settings.mv_fw_spd_3+delta,0,settings.mv_fw_dur)    #+x
        # if action ==1:
        #     start, duration = self.move_forward_Speed(-settings.mv_fw_spd_3+delta,0,settings.mv_fw_dur)   #-x
        # if action ==2:
        #     start, duration = self.move_forward_Speed(0,settings.mv_fw_spd_3+delta,settings.mv_fw_dur)    #+y
        # if action ==3:
        #     start, duration = self.move_forward_Speed(0,-settings.mv_fw_spd_3+delta,settings.mv_fw_dur)   #-y
        print("Action:",action)
        if action == 0:
            start, duration = self.straight(0.5, 0.5)  # move forward #1 #0.5 for change
        if action == 1:
            start, duration = self.yaw_right(25, 0.05)  # yaw right
        if action == 2:
            start, duration = self.yaw_right(-25, 0.05)
        # if action ==3:
        #      self.client.hoverAsync().join()

# yaw left

        collided = (self.client.getMultirotorState().trip_stats.collision_count > 0)
        #collided = self.client.getCollisionInfo().has_collided

        #print("collided----actual:", self.client.getCollisionInfo().has_collided," timestamp:" ,self.client.getCollisionInfo().time_stamp)
        #print("collided " + str(self.client.getMultirotorState().trip_stats.collision_count > 0))

        return collided
