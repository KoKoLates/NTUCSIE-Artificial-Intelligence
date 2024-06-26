# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import math
import util
import random

from game import Agent, AgentState, Directions
from pacman import GameState

from util import manhattanDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves: list = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(
        self,
        currentGameState: GameState, 
        action: str
    ) -> float:
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState: GameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates: list[AgentState] = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        if action == Directions.STOP:
            return -math.inf
        
        new_food_dist: list = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if not len(new_food_dist):
            return math.inf
        
        new_ghost_state: AgentState = newGhostStates[0]
        new_ghost_dist: float = manhattanDistance(newPos, new_ghost_state.getPosition())

        return successorGameState.getScore() + new_ghost_dist / min(new_food_dist)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def terminated(
        self,
        game_state: GameState,
        depth: int
    ) -> bool:
        return depth == self.depth * game_state.getNumAgents() \
            or game_state.isLose() or game_state.isWin()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.get_max_value(gameState, 0, 0)[1]

    def get_value(
        self,
        game_state: GameState,
        agent_index: int, 
        depth: int
    ) -> float:
        if self.terminated(game_state, depth):
            return self.evaluationFunction(game_state)
        
        return self.get_min_value(game_state, agent_index, depth)[0] \
            if agent_index else self.get_max_value(game_state, agent_index, depth)[0]
    
    def get_max_value(
        self, 
        game_state: GameState,
        agent_index: int, depth: int
    ) -> tuple:
        value: float = -math.inf
        action: str = None
        for next_action in game_state.getLegalActions(agent_index):
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()
            next_value: float = self.get_value(next_state, next_agent, depth + 1)

            if next_value > value:
                value, action = next_value, next_action

        return value, action
    
    def get_min_value(
        self,
        game_state: GameState,
        agent_index: int, depth: int
    ) -> tuple:
        value: float = math.inf
        action: str = None
        for next_action in game_state.getLegalActions(agent_index):
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()
            next_value: float = self.get_value(next_state, next_agent, depth + 1)

            if next_value < value:
                value, action = next_value, next_action

        return value, action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """ Your minimax agent with alpha-beta pruning (question 3) """

    def getAction(self, gameState: GameState):
        """ Returns the minimax action using self.depth and self.evaluationFunction """

        "*** YOUR CODE HERE ***"
        return self.get_max_value(gameState, 0, 0, -math.inf, math.inf)[1]

    def get_value(
        self, game_state: GameState,
        agent_index: int, depth: int, alpha: float, beta: float
    ) -> float:
        if self.terminated(game_state, depth):
            return self.evaluationFunction(game_state)
        
        return self.get_min_value(game_state, agent_index, depth, alpha, beta)[0] if agent_index \
            else self.get_max_value(game_state, agent_index, depth, alpha, beta)[0]
    
    def get_max_value(
        self, game_state: GameState,
        agent_index: int, depth: int, alpha: float, beta: float
    ) -> tuple[float, str]:
        value: float = -math.inf
        action: str = None
        for next_action in game_state.getLegalActions(agent_index):
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()
            next_value: float = self.get_value(next_state, next_agent, depth + 1, alpha, beta)

            if next_value > value:
                value, action = next_value, next_action

            if value > beta: break
            alpha = max(alpha, value)

        return value, action

    def get_min_value(
        self, game_state: GameState,
        agent_index: int, depth: int, alpha: float, beta: float
    ) -> tuple[float, str]:
        value: float = math.inf
        action: str = None
        for next_action in game_state.getLegalActions(agent_index):
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()
            next_value: float = self.get_value(next_state, next_agent, depth + 1, alpha, beta)

            if next_value < value:
                value, action = next_value, next_action

            if value < alpha: break
            beta = min(beta, value)

        return value, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """ Your expectimax agent (question 4) """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        "*** YOUR CODE HERE ***"
        return self.get_max_value(gameState, 0, 0)[1]
    
    def get_value(
        self,
        game_state: GameState,
        agent_index: int,
        depth: int
    ) -> float:
        if self.terminated(game_state, depth):
            return self.evaluationFunction(game_state)
        
        return self.get_expect_value(game_state, agent_index, depth) \
            if agent_index else self.get_max_value(game_state, agent_index, depth)[0]
        
    def get_max_value(
        self,
        game_state: GameState,
        agent_index: int,
        depth: int
    ) -> tuple[float, str]:
        value: float = -math.inf
        action: str = None

        for next_action in game_state.getLegalActions(agent_index):
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()

            next_value: float = self.get_value(next_state, next_agent, depth + 1)
            if next_value > value:
                value, action = next_value, next_action

        return value, action

    def get_expect_value(
        self,
        game_state: GameState,
        agent_index: int,
        depth: int
    ) -> float:
        value: float = 0
        legal_action: str = game_state.getLegalActions(agent_index)

        for next_action in legal_action:
            next_state: GameState = game_state.generateSuccessor(agent_index, next_action)
            next_agent: int = (agent_index + 1) % game_state.getNumAgents()

            value += self.get_value(next_state, next_agent, depth + 1) / len(legal_action)

        return value


def scoreEvaluationFunction(currentGameState: GameState) -> float:
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    weights: list[int] = [-15, -2, 1, 3, 150] ## weight modification here
    agent_pose: tuple = currentGameState.getPacmanPosition()

    ## food score
    food_dist: list = [manhattanDistance(agent_pose, pose) \
                       for pose in currentGameState.getFood().asList()]
    food_cnts: int = len(food_dist)
    food_dist_min: float = min(food_dist) if food_cnts else 0
    food_score: float = weights[0] * food_cnts + weights[1] * food_dist_min

    ## ghost score
    ghost_score: float = 0
    ghost_dist_min: float = math.inf
    ghost_states: list[AgentState] = [ghost for ghost in currentGameState.getGhostStates()]
    for ghost_state in ghost_states:
        ghost_dist: float = manhattanDistance(agent_pose, ghost_state.getPosition())
        if not ghost_state.scaredTimer:
            ghost_dist_min = min(ghost_dist, ghost_dist_min)
        elif ghost_state.scaredTimer > ghost_dist:
            ghost_score += weights[4] - ghost_dist

    ghost_score += ghost_dist
    ghost_score *= weights[2]

    ## capsule score
    capsule_cnts: int = len(currentGameState.getCapsules())
    capsule_score: float = weights[3] * capsule_cnts

    return currentGameState.getScore() + ghost_score + food_score + capsule_score

# Abbreviation
better = betterEvaluationFunction

