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


from maze import Maze

def search(maze:Maze, searchMethod:str):
    return {
        'bfs': bfs,
        'astar': astar,
        'astar_corner': astar_corner,
        'astar_multi': astar_multi,
        'fast': fast
    }.get(searchMethod)(maze)

def bfs(maze: Maze) -> list[tuple]:
    """
    Breadth First Search algorithms.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    @rtype: list[tuple]
    """
    return searching(maze, Queue())

def astar(maze: Maze) -> list[tuple]:
    """
    A* Search algorithms.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordniates of each state in the computed path.
    """
    return searching(maze, PriorityQueue(), heuristic_single)

def astar_corner(maze:Maze):
    """
    A* searching for corner target problem.
    """
    return searching(maze, PriorityQueue(), heuristic_multiple)


def astar_multi(maze):
    """
    """
    # TODO: Write your code here
    return []

def fast(maze):
    """
    """
    # TODO: Write your code here
    return []
    
class Node(object):
    def __init__(self, state: tuple[tuple, set], path: list, priority: int=0) -> None:
        """ node object store the information for current position
        @param state: a tuple store the current pose and unvisited target
        @param path: the path from the root node to current node
        @param priority: the priority weight of the nod
        """
        self.state: tuple[tuple, set] = state
        self.path: list = path
        self.priority: int = priority

    def __str__(self) -> str:
        """ print out the information of the node
        @return: the node information
        @rtype: str
        """
        return f"State: pose -> {self.state[0]} unvisited target -> {self.state[1]}\n\
            Path Length: {len(self.path)} | Priority: {self.priority}"

class Queue(object):
    def __init__(self) -> None:
        self.container: list[Node] = []

    def push(self, node: Node) -> None:
        """ insert the node into the container at head
        @param node: the insert node
        """
        self.container.insert(0, node)
    
    def pop(self) -> Node:
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

    def push(self, node: Node) -> None:
        """ insert a node into the priority queue
        @param node: the node to be insert
        """
        if self.isEmpty():
            self.container.append(node)
            return
        
        for index in range(len(self.container)):
            if node.priority >= self.container[index].priority:
                self.container.insert(index, node)
                break

            else:
                if index == len(self.container) - 1:
                    self.container.append(node)

def searching(maze: Maze, container: Queue | PriorityQueue, heuristic=None) -> list:
    """
    A state representation model for searching algorithms.
    @param maze: The maze problem for searching.
    @param container: The date strcuture storing the node and access with priority.
    @param heuristic: The heuristic function using for searching.

    @return: The path from start point to target.
    @rtype: list
    """
    start: tuple = maze.getStart()
    state: tuple[tuple, list] = (start, maze.getObjectives())
    if maze.isObjective(start[0], start[1]):
        state[1].remove(start)

    container.push(Node(
        state, [start], 
        heuristic(state) if heuristic else None
    ))

    visited: list[str] = [] # create a list to record the state that has been visited
    while not container.isEmpty():
        node: Node = container.pop()
        pose, unvisited_target = node.state

        if not len(unvisited_target): # all target has been visited
            return node.path            
            
        for n in maze.getNeighbors(pose[0], pose[1]):
            next_state: tuple[tuple, list] = (n, unvisited_target)
            print(n, unvisited_target)
            if str(next_state) not in visited:
                visited.append(str(next_state))
                if maze.isObjective(n[0], n[1]):
                    next_state[1].remove(n)
                container.push(Node(
                    next_state, node.path + [n],
                    heuristic(next_state) + len(node.path) if heuristic else None
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
    """
    """
    pose, unvisitied = state
    return manhattan(pose, unvisitied[0])

def heuristic_multiple(state: tuple[tuple, list]) -> int:
    """
    """
    pose, unvisited_target = state
    print(pose, unvisited_target)
    if not len(unvisited_target):
        return 0
    
    closest_target: tuple = unvisited_target[0]
    closest_distance: int = manhattan(pose, unvisited_target[0])
    if len(unvisited_target) == 1:
        return closest_distance

    for target in unvisited_target[1:]:
        distance = manhattan(pose, target)
        if distance < closest_distance:
            closest_target = target
            closest_distance = distance

        
        
        farest_distance = 0
        for target in unvisited_target:
            farest_distance = max(farest_distance, manhattan(closest_target, target))

        return closest_distance + farest_distance




def heuristic_corner(maze:Maze, pose:tuple, parent:Node=None) -> int:
    """
    The heuristic function for corner target path planning.
    @param maze: The problem maze.
    @param pose: The current pose.
    @param heuristic: The heuristic of uniform cost.

    @return: The current pose heuristic metric.
    @rtpye: int
    """
    def manhattan_distance(pose1:tuple, pose2:tuple) -> int:
        return abs(pose1[0] - pose2[0]) + abs(pose1[1] - pose2[1])
    
    if maze.isObjective(pose[0], pose[1]):
        return 0
    
    target: list = maze.getObjectives()
    unvisited_corner = list(set(target) - set(target).intersection(set(parent.path))) \
        if parent else list(set(target))
    
    distance:list = []
    for corner in unvisited_corner:
        distance.append(manhattan_distance(pose, corner))

    return max(distance)

    # g = len(parent.path) if parent else 0    
    # target: list = maze.getObjectives()
    # unvisited_corner = list(set(target) - set(target).intersection(set(parent.path))) \
    #     if parent else list(set(target) - set(pose))
    # previous_pose: tuple = pose
    # heuristic: int = 0

    # while unvisited_corner:
    #     closest_corner: tuple = unvisited_corner[0]
    #     closest_corner_distance: int = manhattan_distance(previous_pose, closest_corner)

    #     for i in unvisited_corner[1:]:
    #         distance = manhattan_distance(previous_pose, i)
    #         if distance < closest_corner_distance:
    #             closest_corner = i
    #             closest_corner_distance = distance

    #     heuristic += closest_corner_distance
    #     unvisited_corner.remove(closest_corner)
    #     previous_pose = closest_corner

    # return heuristic
    
def astar_test(maze: Maze, container: PriorityQueue) -> list[tuple]:
    """
    Testing on A* searching algorithm to fin out a transition model
    """
    def huristic(state):
        pose, unvisited_target = state
        if not len(unvisited_target):
            return 0
        
        import math
        def manhattan(pose1: tuple, pose2: tuple) -> int:
            return abs(pose1[0] - pose2[0]) + abs(pose1[1] - pose2[1])

        cloest_distance, cloest_target = math.inf, None
        for t in unvisited_target:
            if manhattan(pose, t) < cloest_distance:
                cloest_distance = manhattan(pose, t)
                cloest_target = t

        if len(unvisited_target) == 1:
            return cloest_distance
        
        max_distance = 0
        for t in unvisited_target:
            max_distance = max(max_distance, manhattan(cloest_target, t))
            
        return cloest_distance + max_distance
    

    ## model start here
    start: tuple = maze.getStart()
    state: tuple = (start, set(maze.getObjectives()))
    if maze.isObjective(start[0], start[1]):
        state[1].discard(start)

    container.push(Node(start, [start], huristic(state)))
    closed: set[str] = set()

    while not container.isEmpty():
        node: Node = container.pop()
        pose, unvisited_target = state

        if not len(unvisited_target):
            return node.path
        
        if str(state) not in closed:
            closed.add(str(state))
            for n in maze.getNeighbors(pose[0], pose[1]):
                next_unvisited_target = unvisited_target.copy()
                next_state = (n, next_unvisited_target)
                if maze.isObjective(n[0], n[1]):
                    next_state[1].discard(n)

                next_path = node.path.copy()
                next_path.append(n)

                container.push(next_state, next_path, len(next_path) - 1 + huristic(next_state))

    return []
