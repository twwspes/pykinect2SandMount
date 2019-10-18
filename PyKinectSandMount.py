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

    didBackgroundDepthSaved = False
    didBackgroundDepthLoaded = False
    arrayOfBackgroundDepth = []

    def __init__(self):
        pygame.init()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

        # back buffer surface for getting Kinect infrared frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

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
            self.didBackgroundDepthLoaded = True

    def draw_depth_SandMount_frame(self, frame, target_surface):
        self.load_depth_frame()
        target_surface.lock()
        objectHeights = self.arrayOfBackgroundDepth - frame
        frame8bit = np.zeros([self._kinect.depth_frame_desc.Width * self._kinect.depth_frame_desc.Height, 3 ], dtype = np.uint8)
        for i in range(self._kinect.depth_frame_desc.Width * self._kinect.depth_frame_desc.Height):
            if objectHeights[i] <= 0:
                frame8bit[i] = np.uint8([255,255,255])
            elif objectHeights[i] <= 40:
                frame8bit[i] = np.uint8([0,255,0])
            elif objectHeights[i] <= 80:
                frame8bit[i] = np.uint8([0,0,255])
            else:
                frame8bit[i] = np.uint8([255,0,0])
        # f8 = np.uint8(frame.clip(1,4000) / 16.)
        # frame8bit = np.dstack((f8, f8, f8))
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

