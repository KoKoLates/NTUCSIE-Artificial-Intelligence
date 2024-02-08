# hw1.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This file contains the main application that is run for this homework. It
initializes the pygame context, and handles the interface between the
game and the search algorithm.
"""

import time
import pygame
import argparse

from maze import Maze
from agent import Agent
from search import search
from pygame.locals import *

class Application(object):
    def __init__(
            self, human:bool=True, scale:int=20, 
            fps:int=30, alt_color:bool=False
        ) -> None:
        self.running = True
        self.displaySurface = None
        self.scale = scale
        self.fps = fps
        self.windowTitle = 'HW1: '
        self.alt_color = alt_color
        self.__human = human
    
    def initialize(self, filename:str) -> None:
        """
        Initialize the application
        @param filename: the filename of the maze
        """
        self.windowTitle += filename

        self.maze = Maze(filename)
        self.gridDim = self.maze.getDimensions()

        self.windowHeight = self.gridDim[0] * self.scale
        self.windowWidth = self.gridDim[1] * self.scale

        self.blockSizeX = int(self.windowWidth / self.gridDim[1])
        self.blockSizeY = int(self.windowHeight / self.gridDim[0])

        if not self.__human:
            return
        
        self.agentRadius = min(self.blockSizeX, self.blockSizeY) / 4
        self.agent = Agent(self.maze.getStart(), self.maze, self.blockSizeX, self.blockSizeY)
            
    def execute(
            self, filename:str, 
            searchMethod:str, save:str
        ) -> None:
        """ Execute the application for searching in the maze
        @param filename: the filename of the maze problem
        @param searchMethod: the name of searching algorithms
        @param save: the flag for saving the output image
        """
        self.initialize(filename)

        if self.maze is None:
            print('No maze created')
            raise SystemExit
        
        if not self.__human:
            t1 = time.time()
            path = search(self.maze, searchMethod)
            total_time = time.time() - t1 # time in seconds
            statesExplored = self.maze.getStatesExplored()
        else:
            path, statesExplored = [], 0

        pygame.init()
        self.displaySurface = pygame.display.set_mode((self.windowWidth, self.windowHeight), 
                                                      pygame.HWSURFACE)
        self.displaySurface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)

        if self.__human:
            self.drawPlayer()
        else:
            print(f'[Results]\nPath Length: {len(path)}' + 
                  f' | State Explored: {statesExplored}' +
                  f' | Total Time: {total_time:.5f} sec' + 
                  f' | Validation: {self.maze.isValidPath(path)}')
            self.drawPath(path)
        
        self.drawMaze()
        self.drawStart()
        self.drawObjective()

        pygame.display.flip()
        if save is not None:
            pygame.image.save(self.displaySurface, save)
            self.running = False
        
        clock = pygame.time.Clock()
        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            clock.tick(self.fps)

            if keys[K_ESCAPE]:
                raise SystemExit
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                
            if not self.__human:
                continue

            key_binds: dict = {
                K_UP:   self.agent.moveUp,
                K_DOWN: self.agent.moveDown,
                K_RIGHT:self.agent.moveRight,
                K_LEFT: self.agent.moveLeft
            }

            for key, movement in key_binds.items():
                if keys[key]:
                    movement()
            
            self.gameLoop()
                
    def gameLoop(self):
        self.drawObjective()
        self.drawPlayer()
        self.agent.update()
        pygame.display.flip()
    
    def getColor(self, pathLength:int, index:int, alt_color:bool=False) -> tuple:
        start_color = (64, 224, 208) if alt_color else (255, 0, 0)
        end_color = (139, 0, 139) if alt_color else (0, 255, 0)

        r_step = (end_color[0] - start_color[0]) / pathLength
        g_step = (end_color[1] - start_color[1]) / pathLength
        b_step = (end_color[2] - start_color[2]) / pathLength

        r = start_color[0] + index * r_step
        g = start_color[1] + index * g_step
        b = start_color[2] + index * b_step
        return (r, g, b)

    def drawPath(self, path:list) -> None:
        for p in range(len(path)):
            color = self.getColor(len(path), p, self.alt_color)
            self.drawSquare(path[p][0], path[p][1], color)

    def drawWall(self, row:int, col:int) -> None:
        pygame.draw.rect(
            self.displaySurface, (0, 0, 0), 
            (
                col * self.blockSizeX, 
                row * self.blockSizeY, 
                self.blockSizeX, 
                self.blockSizeY
            ), 0
        )
        
    def drawCircle(self, row:int, col:int, color, radius=None) -> None:
        if radius is None:
            radius = min(self.blockSizeX, self.blockSizeY) / 4
        pygame.draw.circle(
            self.displaySurface, color, 
            (
                int(col * self.blockSizeX + self.blockSizeX / 2), 
                int(row * self.blockSizeY + self.blockSizeY / 2)
            ), 
            int(radius)
        )

    def drawSquare(self, row:int, col:int, color) -> None:
        pygame.draw.rect(
            self.displaySurface, color, 
            (
                col * self.blockSizeX, 
                row * self.blockSizeY, 
                self.blockSizeX, 
                self.blockSizeY
            ), 0
        )

    def drawPlayer(self) -> None:
        if self.agent.lastRow is not None and self.agent.lastCol is not None:
            self.drawCircle(self.agent.lastRow, self.agent.lastCol, (0, 0, 255))
        self.drawCircle(self.agent.row, self.agent.col, self.agent.color)

    def drawObjective(self) -> None:
        for obj in self.maze.getObjectives():
            self.drawCircle(obj[0], obj[1], (0, 0, 0))

    def drawStart(self) -> None:
        row, col = self.maze.getStart()
        pygame.draw.rect(
            self.displaySurface, (0, 0, 255), 
            (
                int(col * self.blockSizeX + self.blockSizeX / 4), 
                int(row * self.blockSizeY + self.blockSizeY / 4), 
                int(self.blockSizeX * 0.5), 
                int(self.blockSizeY * 0.5)
            ), 0
        )
    
    def drawMaze(self) -> None:
        for row in range(self.gridDim[0]):
            for col in range(self.gridDim[1]):
                if self.maze.isWall(row, col):
                    self.drawWall(row, col)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW1 Search')

    parser.add_argument('filename', help='path to maze file [REQUIRED]')
    parser.add_argument('--method', dest='search', type=str, default='bfs',
                        choices=['bfs', 'astar', 'astar_corner', 'astar_multi', 'fast'], 
                        help='search method -- default: bfs')
    parser.add_argument('--scale', dest='scale', type=int, default=20, 
                        help='scale -- default: 20')
    parser.add_argument('--fps', dest='fps', type=int, default=30, 
                        help='fps for the display -- default: 30')
    parser.add_argument('--human', default=False, action='store_true', 
                        help='flag for human playable -- default: False')
    parser.add_argument('--save', dest='save', type=str, default=None, 
                        help='save output to image file -- default: not save')
    parser.add_argument('--altcolor', dest='altcolor', default='False', action='store_true', 
                        help='view in an alternate color scheme')
    args = parser.parse_args()

    app = Application(args.human, args.scale, args.fps, args.altcolor)
    app.execute(args.filename, args.search, args.save)
