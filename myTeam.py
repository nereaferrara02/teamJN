# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import os
import json

from captureAgents import CaptureAgent
from game import *
from util import nearestPoint



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ApproximativeQAgent(CaptureAgent):
    """
    A base class for approximative q-learning agents
    """
    def __init__(self, index, time_for_computing=.1, alpha=0.2, epsilon=0.2, discount=0.6, Ntraining = 100):
        super().__init__(index, time_for_computing)
        self.start = None
        self.weights = util.Counter()
        self.alpha = alpha 
        self.discount = discount  
        self.epsilon = epsilon  
        self.num_training = Ntraining
        self.update_count = 0
        self.load_file_weights() #load the file with the updated weights
         
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    #FUNCTIONS TO LOAD AND STORE THE UPDATED WEIGHTS INTO A FILE
    #IN THIS WAY THE OFFENSIVE AGENT WILL LEARN AT EACH GAME FROM THE UPDATED WEIGHTS 
    def load_file_weights(self):
        weights_file = 'weights_offensive_teamJN.json'
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as file:
                self.weights = json.loads(file.read())
        else:
            self.weights = self.initialize_weights()
            self.save_weights()

    def save_weights(self):
        weights_file = 'weights_offensive_teamJN.json'
        with open(weights_file, 'w') as file:
            file.write(json.dumps(self.weights)) 
            
   
    #CHOOSE THE BEST ACTION  
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        actions.remove(Directions.STOP)
        action = None

        if len(actions) == 0:
            action = None
    
        if util.flipCoin(self.epsilon): 
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(game_state)

            for action in actions:
                next_state = self.get_successor(game_state, action)
                reward = self.get_reward(game_state, next_state,action)
                self.update_weights(game_state, action, reward)  
                self.save_weights()
        return action 
    
      
    #UPDATE WEIGHTS ACCORDING APPROXIMATIVE Q-LEARNING
    def update_weights(self, game_state, action, reward):
        """
        Updated the weights according the formula of approximate Q-learning
        """
        features = self.get_features(game_state, action)
        next_state = self.get_successor(game_state, action)

        correction = (reward + self.discount*self.get_value(next_state)) - self.evaluate_function(game_state, action)
        for feature in features:
            self.weights[feature] += self.alpha*correction*features[feature]
        
        self.update_count += 1
    
    #FUNCTIONS REGARDING COMPUTE Q-VALUES (WEIGHTS)
    def computeValueFromQValues(self, state):
        qValues = []
        legal_actions = state.get_legal_actions(self.index)
        
        if len(legal_actions) == 0:
            return 0.0
        
        qValues = [self.evaluate_function(state, action) for action in legal_actions]
        maxQvalue = max(qValues)
        
        return maxQvalue
    
    def get_value(self, state):
        return self.computeValueFromQValues(state)

    def computeActionFromQValues(self, state):
        legal_actions = state.get_legal_actions(self.index)
        legal_actions.remove(Directions.STOP) #To avoid the pacman stops

        if not legal_actions:
            return None

        qValues = [self.evaluate_function(state, action) for action in legal_actions]
        maxQvalue = max(qValues)
        best_actions = [a for a, q in zip(legal_actions, qValues) if q == maxQvalue]
        
        if best_actions:
            best_action = random.choice(best_actions)
            return best_action
        else:
            # If best_actions is empty return a random legal action
            return random.choice(legal_actions)

    #FUNCTION TO GET Q-VALUES (WEIGHTS)
    def evaluate_function(self, game_state, action): 
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        return features*self.weights
  
    #THIS FUNCTIONS WILL BE OVERRIDE BY THE OFFENSIVE AGENT  
    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features        

    def get_reward(self, game_state, next_state, action):
        return 0
    
    #OTHER FUNCTIONS
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
     
    def initialize_weights(self):
        return {
            "successor_score": random.randint(0,1), 
            "food_left": random.randint(0,1), 
            "distance_to_food": random.randint(0,1)
        }         

    def final(self, state):
        "Called at the end of each game."
        CaptureAgent.final(self, state)

        #if the agent finish his training we print the best weights
        if self.update_count == self.num_training: 
            print(self.weights)

class OffensiveQAgent(ApproximativeQAgent):
    """
    An approximative q-learning agent that seeks food. 
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = game_state.get_score()
        features['food_left'] = -len(food_list)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        return features
    
    #REWARD FUNCTION   
    def get_reward(self, game_state, next_state, action):
        """
        This function calculates the reward of the offensive agent based on some factors
        """
        reward = 0
        my_pos = game_state.get_agent_position(self.index)
        n_current_food = len(self.get_food(game_state).as_list())
        current_food = self.get_food(game_state).as_list()
        
        #Positive reward if the agent (pacman) eats the opponent food 
        distance_to_food = self.get_min_distance(my_pos, current_food)
        if distance_to_food == 1:
            n_next_food = len(self.get_food(next_state).as_list())
            if n_next_food < n_current_food:
                reward = 10
        
        if self.close_ghosts(game_state):
            reward = - 5 #if the agent is close to a ghost we give a negative reward
            next_my_pos = next_state.get_agent_state(self.index).get_position()
            if next_my_pos == self.start:
                reward = -100 #if the ghost kill us and we return to the base we give a negative reward

        return reward 
    
    #AUXILIAR FUNCTIONS 
    def get_min_distance(self, my_pos, positions): 
        """
        This function calculates the min distance between two positions
        """
        return min([self.get_maze_distance(my_pos, pos) for pos in positions])

    def close_ghosts(self, game_state):
        """
        This function return True or False based on if the pacman 
        is close to the ghost where the distance between them is less then 2
        """
        my_pos = game_state.get_agent_position(self.index)
        ghosts = self.get_opponents(game_state)

        for ghost in ghosts:
            ghost_state = game_state.get_agent_state(ghost)
            if ghost_state is not None:
                ghost_pos = game_state.get_agent_position(self.index)
                if self.get_maze_distance(my_pos, ghost_pos) < 2: 
                    return True  
        return False  


class ReflexAgent(CaptureAgent): 
    """
    A base class for reflex agents that choose score-maximizing actions. 
    This code remain unchanged from the team_name_1
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    #THIS FUNCTIONS WILL BE OVERRIDE BY THE DEFENSIVE AGENT
    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}
    
class DefensiveReflexAgent(ReflexAgent):
    """
    A reflex agent that keeps its side Pacman-free. 
    """

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state of defensive agent
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1 
        
        # Detects if our food is being eaten by the enemy 
        food_eaten = self.get_food_you_are_defending(game_state).count() > self.get_food_you_are_defending(successor).count()
        features['food_eaten'] = int(food_eaten)
            
        # If the agent detects the incoming enemy, the agent will follow the invading enemy only if is not scared
        features['follow_enemy'] = 0
        if len(invaders) > 0 and not successor.get_agent_state(self.index).scared_timer:
            features['follow_enemy'] = 1
    
        return features

    def get_weights(self, game_state, action):
        """
        Returns weights based the significance of the features
        """
        return {'num_invaders': -1000, 'on_defense': 300, 'invader_distance': -10, 'stop': -500, 'reverse': -2, 'food_eaten': -100, 'follow_enemy': 100}

    
