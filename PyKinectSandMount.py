from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys
import numpy as np


if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                    pygame.color.THECOLORS["blue"], 
                    pygame.color.THECOLORS["green"],
                    pygame.color.THECOLORS["orange"], 
                    pygame.color.THECOLORS["purple"], 
                    pygame.color.THECOLORS["yellow"], 
                    pygame.color.THECOLORS["violet"]]


class SandMountRuntime(object):

    didBackgroundDepthSaved = True
    didBackgroundDepthLoaded = False
    arrayOfBackgroundDepth = []
    arrayOfWaterDrop = []
    limitedOffsetX = 54
    limitedOffsetY = 63
    limitedWidth = 336
    limitedHeight = 189

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
        self._frame_surface = pygame.Surface((int(1920/self.limitedWidth)*self.limitedWidth, int(1080/self.limitedHeight)*self.limitedHeight), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((int(1920/self.limitedWidth)*self.limitedWidth, int(1080/self.limitedHeight)*self.limitedHeight), 
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
            print("arrayOfBackgroundDepth")
            print(self.arrayOfBackgroundDepth)
            f.close()
            self.didBackgroundDepthLoaded = True

    def draw_depth_SandMount_frame(self, frame, target_surface):
        self.load_depth_frame()
        target_surface.lock()
        # get an array of heights of any objects on top of the background, and flipping its left and right sides
        objectHeights = self.arrayOfBackgroundDepth - frame + 70
        objectHeights = np.reshape(objectHeights, (self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
        objectHeights = np.fliplr(objectHeights)
        objectHeights = objectHeights[self.limitedOffsetY:(self.limitedOffsetY + self.limitedHeight), self.limitedOffsetX:(self.limitedOffsetX + self.limitedWidth)]
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
        frame8bit = np.dstack((f8Blueint, f8Greenint, f8Redint))
        # f8Red = np.reshape(f8Red, (self.limitedHeight * self.limitedWidth, ))
        # f8Green = np.reshape(f8Green, (self.limitedHeight * self.limitedWidth, ))
        # f8Blue = np.reshape(f8Blue, (self.limitedHeight * self.limitedWidth,))
        # frame8bit = np.dstack((f8Blue, f8Green, f8Red))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
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
                        self._screen = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN, 32)
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


__main__ = "Kinect v2 InfraRed"
game =SandMountRuntime()
game.run()

