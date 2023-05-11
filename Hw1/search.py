
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
    return searching(maze, PriorityQueue(), heuristic_corner)


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
    """ Node object store the information of position in the map """
    def __init__(self, pose:tuple, path:list=[], priority:int=0) -> None:
        self.pose: tuple = pose
        self.path: list = path
        self.priority: int = priority

class Queue(object):
    def __init__(self) -> None:
        self.queue: list = []

    def push(self, node:Node) -> None:
        self.queue.insert(0, node)

    def pop(self) -> Node:
        return self.queue.pop() if not self.isEmpty() else None
    
    def clear(self) -> None:
        self.queue.clear()

    def isEmpty(self) -> bool:
        return len(self.queue) == 0
    
class PriorityQueue(object):
    def __init__(self) -> None:
        self.heap: list = []

    def push(self, node:Node) -> None:
        if self.isEmpty():
            self.heap.append(node)
            return
        
        for idx in range(len(self.heap)):
            if node.priority >= self.heap[idx].priority:
                self.heap.insert(idx, node)
                break
            else:
                if idx == len(self.heap) - 1:
                    self.heap.append(node)

    def pop(self) -> Node:
        return self.heap.pop() if not self.isEmpty() else None
    
    def clear(self) -> None:
        self.heap.clear()

    def isEmpty(self) -> bool:
        return len(self.heap) == 0

def searching(maze: Maze, container, heuristic=None) -> list:
    """
    A state representation model for searching algorithms.
    @param maze: The maze problem for searching.
    @param container: The date strcuture storing the node and access with priority.
    @param heuristic: The heuristic function using for searching.

    @return: The path from start point to target.
    @rtype: list
    """
    start: tuple = maze.getStart()
    target: list = maze.getObjectives()
    if isinstance(container, Queue):
        container.push(Node(start, [start]))
    else: # Priority Queue for A* search
        container.push(Node(start, [start], heuristic(maze, start)))
    
    visited: list = []
    while not container.isEmpty():
        node: Node = container.pop()
        print(node.pose)
        print(node.path)
        print(target)
        if node.pose not in visited:
            visited.append(node.pose)
        else:
            continue

        if node.pose in target:
            target = list(set(target) - set(node.pose))
            if not target:
                return node.path
            
            else:
                visited.clear()
                continue
 
        for n in maze.getNeighbors(node.pose[0], node.pose[1]):
            if isinstance(container, Queue):
                container.push(Node(n, node.path + [n]))
            else: # Priority Queue for A* search
                container.push(Node(n, node.path + [n], heuristic(maze, n, node)))
       
    return []

def heuristic_single(maze:Maze, pose:tuple, parent:Node=None) -> int:
    """
    Heuristic function design for single target finding.
    The classic A* searching heuristic combine the uniform cost and manhattan distance.
    @param maze: The problem maze
    @param pose: The current pose in the problem maze
    @param huristic: The parent node uniform cost

    @return: The heuristuc in the current pose
    @rtype: int
    """
    target = maze.getObjectives()[0]
    g = len(parent.path) if parent else 0
    h = abs(pose[0] - target[0]) + abs(pose[1] - target[1])
    return g + h


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
    
