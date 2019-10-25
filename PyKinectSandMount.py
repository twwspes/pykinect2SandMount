from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
# generate random integer values
from random import seed
from random import randint

import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import cv2



if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

waterDrops = []

class WaterDrop(object):

    _position = np.array([70, 220])
    _velocity = np.array([0, 0], float)
    _acceleration = np.array([0, 0], float)
    _mass = 1

    def __init__(self, x, y, mass):
        self._position = np.array([y, x])
        self._mass = mass
        self._velocity = np.array([0, 0], float)
        self._acceleration = np.array([0, 0], float)

    def applyForce(self, force):
        f = force / self._mass
        self._acceleration += f

    def update(self):
        self._velocity += self._acceleration
        self._position += self._velocity.astype(int)
        self._acceleration = np.array([0,0], float)

    def setPosition(self, newPosition):
        self._position = newPosition

    def getPosition(self):
        return self._position

    def getVelocity(self):
        return self._velocity

    def checkEdge(self):
        reverseVelocity = False
        if self._position[1] > 270:
            self._position[1] = 270
            reverseVelocity = True
        elif self._position[1] < 70:
            self._position[1] = 70
            reverseVelocity = True
        if self._position[0] > 165:
            self._position[0] = 165
            reverseVelocity = True
        elif self._position[0] < 40:
            self._position[0] = 40
            reverseVelocity = True
        if reverseVelocity == True:
            self._velocity *= -0.1

class SandMountRuntime(object):

    didBackgroundDepthSaved = True
    didBackgroundDepthLoaded = False
    arrayOfBackgroundDepth = []
    limitedOffsetX = 54
    limitedOffsetY = 63
    limitedWidth = 336
    limitedHeight = 189
    arrayOfWaterDrop = np.zeros((limitedHeight, limitedWidth), dtype=np.uint8)

    def __init__(self):
        pygame.init()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

        # back buffer surface for getting Kinect infrared frames, 8bit grey, width and height equal to the Kinect color frame size
        # self._frame_surface = pygame.Surface((self.limitedWidth, self.limitedHeight), 0, 24)
        self._frame_surface = pygame.Surface((int(1920/self.limitedWidth)*self.limitedWidth, int(1080/self.limitedHeight)*self.limitedHeight), 0, 32)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        # self._screen = pygame.display.set_mode((int(1920/self.limitedWidth)*self.limitedWidth, int(1080/self.limitedHeight)*self.limitedHeight), 
        #                                         pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, 32)
        self._screen = pygame.display.set_mode((1920, 1080), 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, 32)
        # self._screen = pygame.display.set_mode((self.limitedWidth, self.limitedHeight), 
        #                                         pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 SandMount")

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_depth_frame(self, frame, target_surface):
        self.load_depth_frame()
        target_surface.lock()
        f8 = np.uint8(frame.clip(1,4000) / 16.)
        frame8bit = np.dstack((f8, f8, f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def addWaterDrop(self, objectHeights, frame8bit):
        objectHeightsint = objectHeights.astype(int)
        # for objHeight, cellOfWaterDrop in np.nditer([objectHeightsint, self.arrayOfWaterDrop], op_flags=['readwrite']):
        #     if objHeight > 350 and objHeight < 400:
        #         cellOfWaterDrop[...] +=np.uint8(20)
        self.arrayOfWaterDrop[np.logical_and((objectHeightsint) > 350, (objectHeightsint)< 400)] += np.uint8(1)

        WaterDepthColorBlue = np.uint8((self.arrayOfWaterDrop*10).clip(0,250))
        WaterDepthColorBlue = np.kron(WaterDepthColorBlue, np.ones((int(1080/self.limitedHeight), int(1920/self.limitedWidth))))
        WaterDepthColorBlueint = WaterDepthColorBlue.astype(np.uint8)
        WaterDepthColor = np.zeros((int(1080/self.limitedHeight)*self.limitedHeight, int(1920/self.limitedWidth)*self.limitedWidth), dtype=np.uint8)
        f8Alphaint = np.ones((int(1080/self.limitedHeight)* self.limitedHeight, int(1920/self.limitedWidth) * self.limitedWidth, ), dtype=np.uint8)
        Waterframe8bit = np.dstack((WaterDepthColorBlueint, WaterDepthColor, WaterDepthColor, f8Alphaint))
        frame8bit = np.reshape(frame8bit, (int(1080/self.limitedHeight)* self.limitedHeight, int(1920/self.limitedWidth) * self.limitedWidth, 4))
        WaterCVmat = cv2.addWeighted(frame8bit, 0.5, Waterframe8bit, 0.5, 0)
        WaterCVmat = np.reshape(WaterCVmat, (int(1080/self.limitedHeight)* self.limitedHeight * int(1920/self.limitedWidth) * self.limitedWidth, 4))
        return WaterCVmat

    def moveWaterDrop(self, objectHeights):
        objectHeightsint = objectHeights.astype(int)
        for y in range(40, objectHeights.shape[0]-10):
            for x in range(40, objectHeights.shape[1]-10):
                objHeightsAtWaterDrop = objectHeightsint[y-1:y+2, x-1:x+2]
                waterDepth = self.arrayOfWaterDrop[y-1:y+2, x-1:x+2]
                sumWaterDepth = waterDepth.sum()
                if sumWaterDepth > 0:
                    whichMatrixRemoved = [1,1,1,1,1,1,1,1,1]
                    averageHeight = 0
                    # isMatrixRemoved = False
                    notWaterDropDistributed = True
                    while notWaterDropDistributed:
                        notMatrixRemoved = True
                        sumWhichMatrixRemoved = sum(whichMatrixRemoved)
                        sumObjHeightAtWaterDrop = objHeightsAtWaterDrop.sum()
                        averageHeight = (sumObjHeightAtWaterDrop + sumWaterDepth) / (sumWhichMatrixRemoved if sumWhichMatrixRemoved != 0 else 1)
                        for i in range(0, 9):
                            n = int(i/3)
                            m = int(i%3)
                            if averageHeight < objHeightsAtWaterDrop[n, m]:
                                objHeightsAtWaterDrop[n, m] = 0
                                notMatrixRemoved = False
                                whichMatrixRemoved[i] = 0
                        if notMatrixRemoved:
                            sumWaterDropDistributed = 0
                            for i in range(0, 9):
                                n = int(i/3)
                                m = int(i%3)
                                if whichMatrixRemoved[i] == 1:
                                    self.arrayOfWaterDrop[y+n-1, x+m-1] = np.uint8(averageHeight - objHeightsAtWaterDrop[n, m])
                                    sumWaterDropDistributed += averageHeight - objHeightsAtWaterDrop[n, m]
                                else:
                                    self.arrayOfWaterDrop[y+n-1, x+m-1] = np.uint8(0)
                            remainingWater = sumWaterDepth - sumWaterDropDistributed
                            seed(1)
                            random = randint(0,8)
                            self.arrayOfWaterDrop[y+int(random/3)-1, x+int(random%3)-1] += np.uint8(remainingWater if remainingWater >=0 else 0)
                            notWaterDropDistributed = False



    def addWaterDrop2(self, objectHeights):
        objectHeightsint = objectHeights.astype(int)
        lowestHeightCoors = np.asarray(np.logical_and((objectHeightsint) > 350, (objectHeightsint)< 400)).nonzero()
        listOfCoors = list(zip(lowestHeightCoors[0], lowestHeightCoors[1]))
        for coors in listOfCoors:
            # waterDrops.append(WaterDrop(coors[0].clip(40, 165), coors[1].clip(70, 275), 1))
            waterDrops.append(WaterDrop(240, 70, 1))

    def moveWaterDrop2(self, waterDrop, objectHeights, target_surface, index):

        # Friction = -1 * coefForce * normalForce * velocity
        # print("Velo X : % 3d, Y : % 2d" %(waterDrop.getVelocity()[1], waterDrop.getVelocity()[0]))
        coefFric = 0.01 # coefficient of friction
        normalForce = 40 # normal force, which is perpendicluar to the surface
        frictionMag = coefFric * normalForce
        friction = np.array([waterDrop.getVelocity()[0]*-1, waterDrop.getVelocity()[1]*-1], dtype=float)
        # print("Velo X : % 3d, Y : % 2d" %(waterDrop.getVelocity()[1], waterDrop.getVelocity()[0]))
        normBase = np.linalg.norm(friction, ord=2, axis=0, keepdims=True)
        normBase = 1 if normBase == 0 else normBase
        friction = friction/normBase
        friction *= frictionMag

        # Calculating vector of waterDrop by using objectHeights
        # print("fric X : % 3d, Y : % 2d ; Velo X : % 3d, Y : % 2d" %(friction[1], friction[0], waterDrop.getVelocity()[1], waterDrop.getVelocity()[0]))
        # objectHeightsfloat = np.kron(objectHeights, np.ones((int(1080/self.limitedHeight), int(1920/self.limitedWidth))))
        objectHeightsint = objectHeights.astype(int)
        positionY = waterDrop.getPosition()[0]
        positionX = waterDrop.getPosition()[1]
        # print("position X : % 3d, Y : % 2d" %(positionX, positionY))
        objHeightsAtWaterDrop = objectHeightsint[positionY-1:positionY+2, positionX-1:positionX+2]
        listOfCoors = []
        try:
            lowestHeightCoors = np.where(objHeightsAtWaterDrop == np.amin(objHeightsAtWaterDrop))
            listOfCoors = list(zip(lowestHeightCoors[0], lowestHeightCoors[1]))
        except:
            listOfCoors = ((0,0))

        # set the waterDrop move
        # seed random number generator
        seed(index)
        waterDrop.applyForce(friction)
        waterDrop.applyForce(np.asarray(listOfCoors[randint(0, len(listOfCoors)-1)])-1)
        waterDrop.update()

        # Drawing on the surface
        positionY = waterDrop.getPosition()[0]
        positionX = waterDrop.getPosition()[1]
        centerX = int(positionX * int(1920/self.limitedWidth))
        centerY = int(positionY * int(1080/self.limitedHeight))
        center = [centerX, centerY]
        # print("Position X : % 3d, Y : % 2d; Velo X : % 3d, Y : % 2d; Altitute: % 2d" %(positionX, positionY, waterDrop.getVelocity()[1], waterDrop.getVelocity()[0], np.amin(objHeightsAtWaterDrop)))
        pygame.draw.circle(target_surface, (0, 0, 0), center, 10)

        # if the waterDrop hit against a wall, it will bounce back
        waterDrop.checkEdge()

    def save_depth_frame(self, frame):
        f = open('Background', 'w+b')
        print(frame)
        binary_format = bytearray(frame)
        f.write(binary_format)
        f.close()

    def load_depth_frame(self):
        if self.didBackgroundDepthLoaded == False:
            f=open('Background',"rb")
            self.arrayOfBackgroundDepth = np.frombuffer(f.read(), dtype=np.uint16)
            self.arrayOfBackgroundDepth = self.opencvProcessing(self.arrayOfBackgroundDepth)
            print("arrayOfBackgroundDepth")
            print(self.arrayOfBackgroundDepth)
            f.close()
            self.didBackgroundDepthLoaded = True

    def opencvProcessing(self, objectHeights):
        objectHeights = cv2.morphologyEx(objectHeights, cv2.MORPH_OPEN, np.ones((5,5), np.uint16))
        return objectHeights

    def draw_depth_SandMount_frame(self, frame, target_surface):
        self.load_depth_frame()
        target_surface.lock()
        frame = self.opencvProcessing(frame)
        # get an array of heights (np.uint16) of any objects on top of the background, and flipping its left and right sides
        objectHeights = self.arrayOfBackgroundDepth - frame + 90
        objectHeights = np.reshape(objectHeights, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
        objectHeights = np.fliplr(objectHeights)
        objectHeights = objectHeights[self.limitedOffsetY:(self.limitedOffsetY + self.limitedHeight), self.limitedOffsetX:(self.limitedOffsetX + self.limitedWidth)]
        objectHeights = self.opencvProcessing(objectHeights)
        # if height between 0 50 then from green to yellow, if height between 50 100 then from yellow to red
        f8Red = np.uint8((objectHeights * 50/18).clip(0,250))
        f8Green = np.uint8((objectHeights * -3 + 500).clip(0,250))
        f8Blue = np.uint8((objectHeights.clip(0,1)))
        f8Redfloat = np.kron(f8Red, np.ones((int(1080/self.limitedHeight), int(1920/self.limitedWidth))))
        f8Redint = f8Redfloat.astype(np.uint8)
        f8Greenfloat = np.kron(f8Green, np.ones((int(1080/self.limitedHeight), int(1920/self.limitedWidth))))
        f8Greenint = f8Greenfloat.astype(np.uint8)
        f8Bluefloat = np.kron(f8Blue, np.ones((int(1080/self.limitedHeight), int(1920/self.limitedWidth))))
        f8Blueint = f8Bluefloat.astype(np.uint8)
        f8Redint = np.reshape(f8Redint, (int(1080/self.limitedHeight)* self.limitedHeight * int(1920/self.limitedWidth) * self.limitedWidth, ))
        f8Greenint = np.reshape(f8Greenint, (int(1080/self.limitedHeight)* self.limitedHeight * int(1920/self.limitedWidth) * self.limitedWidth, ))
        f8Blueint = np.reshape(f8Blueint, (int(1080/self.limitedHeight)* self.limitedHeight * int(1920/self.limitedWidth) * self.limitedWidth, ))
        f8Alphaint = np.ones((int(1080/self.limitedHeight)* self.limitedHeight * int(1920/self.limitedWidth) * self.limitedWidth, ), dtype=np.uint8)
        frame8bit = np.dstack((f8Blueint, f8Greenint, f8Redint, f8Alphaint))
        frame8bit = frame8bit[0]
        # f8Red = np.reshape(f8Red, (self.limitedHeight * self.limitedWidth, ))
        # f8Green = np.reshape(f8Green, (self.limitedHeight * self.limitedWidth, ))
        # f8Blue = np.reshape(f8Blue, (self.limitedHeight * self.limitedWidth,))
        # frame8bit = np.dstack((f8Blue, f8Green, f8Red))
        frame8bit = self.addWaterDrop(objectHeights, frame8bit)
        self.moveWaterDrop(objectHeights)
        # self.addWaterDrop2(objectHeights)
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        # for i, waterDrop in enumerate(waterDrops):
        #     self.moveWaterDrop2(waterDrop, objectHeights, target_surface, i)
        # del address
        target_surface.unlock()

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                if event.type == pygame.KEYDOWN: # If user press Keyboard
                    if event.key == pygame.K_q: # If user press Keyboard q
                        #self._screen = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, 32)
                        self._done = True # Flag that we are done so we exit this loop
                    if event.key == pygame.K_f: # If user press Keyboard f
                        for i, waterDrop in enumerate(waterDrops):
                            seed(1)
                            waterDrop.setPosition(np.array([70, 240]))
                    if event.key == pygame.K_s: # If user press Keyboard f
                        waterDrops.append(WaterDrop(240, 60, 1))
                    if event.key == pygame.K_d: # If user press Keyboard f
                        waterDrops.pop()
                    if event.key == pygame.K_w: # If user press Keyboard f
                        waterDrops.clear()
                        np.delete(self.arrayOfWaterDrop, 0)
                    if event.key == pygame.K_a: # If user press Keyboard a
                        self.didBackgroundDepthSaved = False

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Getting frames and drawing  
            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                if self.didBackgroundDepthSaved == True:
                    # self.draw_depth_frame(frame, self._frame_surface)
                    self.draw_depth_SandMount_frame(frame, self._frame_surface)
                if self.didBackgroundDepthSaved == False:
                    self.save_depth_frame(frame)
                    self.didBackgroundDepthSaved = True
                frame = None
                

            self._screen.blit(self._frame_surface, (0,0))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect Sand Mount"
game =SandMountRuntime()
for i in range(2, 3):
    waterDrops.append(WaterDrop(240, 60, 1))
game.run()

