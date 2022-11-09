# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    # Prefer the close exit (+1), risking the cliff (-10)
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    # High discount to encourage a short path, but allow randomness to discourage cliff
    answerDiscount = .1
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # Prefer the distant exit (+10), risking the cliff (-10)
    # High non-zero discount to prevent infinite loops but aim for distant exit
    answerDiscount = .99
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Prefer the distant exit (+10), avoiding the cliff (-10)
    # High non-zero discount to aim for distant exit, noise to discourage cliffwalking
    answerDiscount = 0.99
    answerNoise = 0.3
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = .99
    answerNoise = .2
    answerLivingReward = 10
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = 1
    answerLearningRate = 1
    # We aren't likely enough to stumble upon (right -> right -> right -> ...) within 50 iterations!
    return "NOT POSSIBLE"
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
