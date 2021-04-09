from math import sqrt, log

def ucb(node):
    return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)
