# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import re, copy
from collections import Counter

class Maze(object):
    # Initializes the Maze objet by reading the maze from a file
    def __init__(self, filename:str) -> None:
        self.__filename:str = filename
        self.__wallChar, self.__startChar, self.__objectiveChar = '%', 'P', '.'
        self.__start = None
        self.__objective = []
        self.__states_explored = 0

        with open(filename) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = [list(line.strip('\n')) for line in lines]

        self.rows, self.cols = len(lines), len(lines[0])
        self.mazeRaw = lines

        if (len(self.mazeRaw) != self.rows) or (len(self.mazeRaw[0]) != self.cols):
            print('Maze Dimension incorrect')
            raise SystemExit
        
        for row in range(len(self.mazeRaw)):
            for col in range(len(self.mazeRaw[0])):
                if self.mazeRaw[row][col] == self.__startChar:
                    self.__start = (row, col)
                elif self.mazeRaw[row][col] == self.__objectiveChar:
                    self.__objective.append((row, col))

    def isWall(self, row:int, col:int) -> bool:
        """
        @return: The given postion is the location of a wall or not
        """
        return self.mazeRaw[row][col] == self.__wallChar
    
    def isObjective(self, row:int, col:int) -> bool:
        """
        @return: The given position is the location of an objective or not
        """
        return (row, col) in self.__objective
    
    def getStart(self) -> tuple:
        return self.__start
    
    def setStart(self, start:tuple) -> None:
        self.__start = start

    def getDimensions(self) -> tuple:
        return (self.rows, self.cols)
    
    def getObjectives(self) -> list:
        return copy.deepcopy(self.__objective)

    def getStatesExplored(self) -> int:
        return self.__states_explored 

    def isValidMove(self, row:int, col:int) -> bool:
        return row >= 0 and row < self.rows and col >= 0 \
               and col < self.cols and not self.isWall(row, col)
    
    def getNeighbors(self, row:int, col:int) -> list:
        possibleNeighbor = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ] 
        neighbor = []
        for r, c in possibleNeighbor:
            if self.isValidMove(r, c):
                neighbor.append((r, c))
        self.__states_explored += 1
        return neighbor

    def isValidPath(self, path) -> str:
        if not isinstance(path, list):
            return "Path must be List"
        
        if not len(path):
            return "Path must not be empty"
        
        if not isinstance(path[0], tuple):
            return "Position must be tuple"
        
        if len(path[0]) != 2:
            return "Postion must be (x, y)"
        
        for i in range(1, len(path)):
            currents = path[i]
            previous = path[i - 1]
            dist = abs((previous[1] - currents[1]) + (previous[0] - currents[0]))
            if dist > 1:
                return "Not single hop"
            
        for pos in path:
            if not self.isValidMove(pos[0], pos[1]):
                return "Not Valid Move"
            
        if not set(self.__objective).issubset(set(path)):
            return "Not all goals passed"
        
        if not path[-1] in self.__objective:
            return "Last point is not goal"
        
        if len(set(path)) != len(path):
            c = Counter(path)
            dup_dots = [p for p in set(c.elements()) if c[p] >= 2]
            for p in dup_dots:
                indices = [i for i, dot in enumerate(path) if dot == p]
                is_dup = True
                for i in range(len(indices) - 1):
                    for dot in path[indices[i] + 1: indices[i + 1]]:
                        if self.isObjective(dot[0], dot[1]):
                            is_dup = False
                            break
                
                if is_dup:
                    return "Unnecessary path detected"
                
        return "Valid"
