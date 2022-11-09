# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

# QUESTION 1
class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for iteration in range(self.iterations):
            iterative_value_dict = util.Counter()
            for state in self.mdp.getStates():
                # Terminal states have value=0
                if self.mdp.isTerminal(state):
                    iterative_value_dict[state] = 0
                # Otherwise do value iteration
                else:
                    maxval = float("-inf")
                    possible_actions = self.mdp.getPossibleActions(state)
                    for action in possible_actions:
                        transition_probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        svals = [prob * (self.mdp.getReward(state, action, prob) + self.discount * self.values[nstate]) for nstate, prob in transition_probabilities]
                        value = value + sum(svals)
                        maxval = max(value, maxval)
                    if maxval != float("-inf"):
                        iterative_value_dict[state] = maxval
            self.values = iterative_value_dict

        # I tried so long to get this to work and i cant tell how this is different from my one that works
        # for iteration in range(self.iterations):
        #     states = self.mdp.getStates()
        #     iterative_state_values = util.Counter()
        #     for state in states:
        #         if self.mdp.isTerminal(state):
        #             iterative_state_values[state] = 0
        #         else:
        #             possible_actions = self.mdp.getPossibleActions(state)
        #             maxval = float("-inf")
        #             for action in possible_actions:
        #                 transition_probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        #                 state_value = 0
        #                 # The actual value iteration part
        #                 for nextstate, probability in transition_probabilities:
        #                     state_value += probability * (self.mdp.getReward(state, action, probability)) + (self.discount * self.values[nextstate])
        #                 maxval = max(state_value, maxval)
        #                 if maxval != float("-inf"):
        #                     iterative_state_values[state] = maxval
        #
        #     self.values = iterative_state_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qval = 0
        transition_probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextstate, prob in transition_probabilities:
            qval += prob * (self.mdp.getReward(state, action, nextstate) + self.discount * self.values[nextstate])
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            possible_actions = self.mdp.getPossibleActions(state)
            action_value = {action: self.computeQValueFromValues(state, action) for action in possible_actions}
            best_action = max(action_value, key=action_value.get)
            return best_action



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

