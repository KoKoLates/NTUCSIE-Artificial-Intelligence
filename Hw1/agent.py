import pygame
from maze import Maze

# The agant is only used when a human player is used, and is therefore not annotated much
class Agent(object):
    def __init__(self, pose:tuple, maze:Maze, blockSizeX:int, blockSizeY:int) -> None:
        self.row, self.col = pose[0], pose[1]
        self.lastRow, self.lastCol = None, None
        self.needUpdate = True
        self.color = (255, 0, 0)
        self.maze: Maze = maze
        self.blockSizeX, self.blockSizeY = blockSizeX, blockSizeY

    def update(self) -> None:
        if self.needUpdate:
            self.needUpdate = False
            position = (int(self.col * self.blockSizeX - self.blockSizeX / 2), 
                        int(self.row * self.blockSizeY - self.blockSizeY / 2))
            pygame.display.flip()

    def canMoveRight(self) -> bool:
        return self.maze.isValidMove(self.row, self.col + 1)

    def canMoveLeft(self) -> bool:
        return self.maze.isValidMove(self.row, self.col - 1)
    
    def canMoveUp(self) -> bool:
        return self.maze.isValidMove(self.row - 1, self.col)
    
    def canMoveDown(self) -> bool:
        return self.maze.isValidMove(self.row + 1, self.col)
    
    def moveRight(self) -> None:
        if self.canMoveRight():
            self.lastCol, self.lastRow = self.col, self.row
            self.needUpdate = True
            self.col += 1

    def moveLeft(self) -> None:
        if self.canMoveLeft():
            self.lastCol, self.lastRow = self.col, self.row
            self.needUpdate = True
            self.col -= 1

    def moveUp(self) -> None:
        if self.canMoveUp():
            self.lastCol, self.lastRow = self.col, self.row
            self.needUpdate = True
            self.row -= 1

    def moveDown(self) -> None:
        if self.canMoveDown():
            self.lastCol, self.lastRow = self.col, self.row
            self.needUpdate = True
            self.row += 1
    