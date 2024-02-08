# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import math
import heapq

from maze import Maze
from typing import Callable

def search(maze:Maze, searchMethod:str):
    return {
        'bfs': bfs,
        'astar': astar,
        'astar_corner': astar_corner,
        'astar_multi': astar_multi,
        'fast': fast
    }.get(searchMethod)(maze)

def bfs(maze: Maze) -> list[tuple]:
    """Breadth First Search algorithms.
    this search algorithm is written for part 1.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    @rtype: list[tuple]
    """
    return searching(maze, Queue())

def astar(maze: Maze) -> list[tuple]:
    """A* Search algorithms.
    this search algorithm is written for part 1 that only single target
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    """
    return searching(maze, PriorityQueue(), heuristic_single)

def astar_corner(maze:Maze) -> list[tuple]:
    """A* searching for part 2 of corner target
    @param maze: the maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    """
    return searching(maze, PriorityQueue(), heuristic_multiple)

def astar_multi(maze):
    """A* searching for part 3 of multiple target
    @param maze: the maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    """
    return searching(maze, PriorityQueue(), heuristic_multiple)

def fast(maze):
    """fast searching for part 4 of multiple target
    @param maze: the maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    """
    return searching(maze, PriorityQueue(), heuristic_fast)

class Queue(object):
    def __init__(self) -> None:
        self.container: list[tuple] = []

    def push(self, node: tuple[int, tuple, list]) -> None:
        """ insert the node into the container at head
        @param node: the insert node
        """
        self.container.insert(0, node)
    
    def pop(self) -> tuple[int, tuple, list]:
        """ remove and return the node in the tail of container
        @return: the node in the tail of container
        @rtype: Node
        """
        return self.container.pop() if not self.isEmpty() else None
    
    def isEmpty(self) -> bool:
        """ check the container is empty or not
        @return: the checking result
        @rtype: bool
        """
        return not len(self.container)
    
    def clear(self) -> None:
        """ clear the contents of the container """
        self.container.clear()    
    

class PriorityQueue(Queue):
    def __init__(self) -> None:
        super().__init__()

    def push(self, node: tuple[int, tuple, list]) -> None:
        """ insert a new node (tuple) into the container 
        then using heapq to heap sort the container 
        @param node: the information for the input pose
        """
        heapq.heappush(self.container, node)

    def pop(self) -> tuple[int, tuple, list]:
        """ remove and return the min heuristic node in the container
        @return: the heuristic, state and path of the node
        @rtype: tuple[int, tuple, list]
        """
        return heapq.heappop(self.container)
    

def searching(
        maze: Maze, 
        container: Queue | PriorityQueue, 
        heuristic: Callable=None
    ) -> list[tuple]:
    """ A state representation model for searching algorithms.
    @param maze: The maze problem for searching.
    @param container: The date strcuture storing the node and access with priority.
    @param heuristic: The heuristic function using for searching.

    @return: The path from start point to target.
    @rtype: list[tuple]
    """
    start: tuple = maze.getStart()
    state: tuple[tuple, set] = (start, set(maze.getObjectives()))
    if maze.isObjective(start[0], start[1]):
        state[1].discard(start)

    container.push((
        heuristic(state) if heuristic else None,
        state, [start] 
    ))
    visisted: list[str] = []
    while not container.isEmpty():
        _, state, path = container.pop()
        pose, unvisited_target = state
        if not len(unvisited_target): # all target have been visited
            return path
        
        if str(state) in visisted:
            continue

        visisted.append(str(state))
        for n in maze.getNeighbors(pose[0], pose[1]):
            next_state = (n, unvisited_target.copy())
            if maze.isObjective(n[0], n[1]):
                next_state[1].discard(n)

            container.push((
                heuristic(next_state) + len(path) if heuristic else None,
                next_state, path + [n]
            ))

    return []

def manhattan(pose1: tuple, pose2: tuple) -> int:
    """ Calculate the manhattan distance between two poses
    @param poas1: the first pose
    @param pose2: the other pose
    @return: the manhatta distance between two poses
    @rtype: int
    """
    return abs(pose1[0] - pose2[0]) + abs(pose1[1] - pose2[1])


def heuristic_single(state: tuple[tuple, set]) -> int:
    """ The heuristic function for single target mission 
    the basic concept is to return the manhattan distance between
    current position and target position
    @param state: the information of current pose and unvisited target
    @return: heuristic evaluated at current pose
    @rtype: int
    """
    pose, unvisitied = state
    return manhattan(pose, list(unvisitied)[0]) if unvisitied else 0

def heuristic_multiple(state: tuple[tuple, list]) -> int:
    """ the heuristic function for multiple (corner) target mission
    @param state: the information of current pose and unvisited target
    @return: heuristic evaluated at current pose
    @rtype: int
    """
    pose, unvisitied = state
    if not len(unvisitied):
        return 0
    
    if len(unvisitied) == 1:
        return heuristic_single(state)
    
    min_dist, closest_target = math.inf, None
    for target in unvisitied:
        if manhattan(pose, target) < min_dist:
            min_dist = manhattan(pose, target)
            closest_target = target

    max_dist = max([manhattan(closest_target, target) for target in unvisitied])
    return max_dist + min_dist


def heuristic_fast(state: tuple[tuple, list]) -> int:
    """the heuristic function for multiple (corner) target mission
    @param state: the information of current pose and unvisited target
    @return: heuristic evaluated at current pose
    @rtype: int
    """
    pose, unvisitied = state
    return sum(manhattan(pose, target) for target in unvisitied)
    