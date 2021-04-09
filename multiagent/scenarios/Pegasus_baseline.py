#value function 


from pulp import LpVariable, lpSum, LpInteger, LpStatusOptimal, LpProblem
from pulp import LpMinimize, LpMaximize, LpStatus
import random
from pyomo.environ import *


def GenerateRandomMILP(VARIABLES, CONSTRAINTS, density=0.2,
                       maxObjCoeff=10, maxConsCoeff=10,
                       tightness=2, rand_seed=2):
    random.seed(rand_seed)
    VARIABLES = VARIABLES
    CONSTRAINTS = CONSTRAINTS
    OBJ = dict((i, random.randint(1, maxObjCoeff)) for i in VARIABLES)
    MAT = dict(((i, j), random.randint(1, maxConsCoeff)
    if random.random() <= density else 0)
               for i in CONSTRAINTS for j in VARIABLES)
    RHS = dict((i, random.randint(int(len(VARIABLES) * density * maxConsCoeff / tightness),
                                  int(len(VARIABLES) * density * maxConsCoeff / 1.5)))
               for i in CONSTRAINTS)

    return OBJ, MAT, RHS


# numVars = 40
# numIntVars = 20
# numCons = 20
numVars = 5
numIntVars = 3
numCons = 1
INTVARS = range(numIntVars)
CONVARS = range(numIntVars, numVars)
VARS = range(numVars)
CONS = range(numCons)
# CONS = ["C"+str(i) for i in range(numCons)]
intvars = LpVariable.dicts("x", INTVARS, 0, None, LpInteger)
convars = LpVariable.dicts("y", CONVARS, 0)

# Generate random MILP
# OBJ, MAT, RHS = GenerateRandomMILP(VARS, CONS, rand_seed = 3)
OBJ = [3, 3.5, 3, 6, 7]
MAT = {(0, 0): 6, (0, 1): 5, (0, 2): -4, (0, 3): 2, (0, 4): -7}
RHS = [5]
LpRelaxation = LpProblem("relax", LpMinimize)
LpRelaxation += (lpSum(OBJ[j] * intvars[j] for j in INTVARS)
                 + lpSum(OBJ[j] * convars[j] for j in CONVARS)), "Objective"
for i in CONS:
    LpRelaxation += (lpSum(MAT[i, j] * intvars[j] for j in INTVARS)
                     + lpSum(MAT[i, j] * convars[j] for j in CONVARS)
                     == RHS[i]), i
# Solve the LP relaxation
status = LpRelaxation.solve()
print(LpStatus[status])
for i in INTVARS:
    print(i, intvars[i].varValue)
for i in CONVARS:
    print(i, convars[i].varValue)

intvars = LpVariable.dicts("x", INTVARS, 0, 100, LpInteger)
IpRestriction = LpProblem("relax", LpMinimize)
IpRestriction += lpSum(OBJ[j] * intvars[j] for j in INTVARS), "Objective"
for i in CONS:
    IpRestriction += lpSum(MAT[i, j] * intvars[j] for j in INTVARS) == RHS[i], i
# Solve the LP relaxation
status = IpRestriction.solve()
print(LpStatus[status])
for i in INTVARS:
    print(i, intvars[i].varValue)

max_iters = 1000
listTemp = []
Master = AbstractModel()
Master.intIndices = Set(initialize=INTVARS)
Master.constraintSet = Set(initialize=CONS)
Master.conIndices = Set(initialize=CONVARS)
Master.intPartList = Set(initialize=listTemp)
Master.dualVarSet = Master.constraintSet * Master.intPartList
Master.theta = Var(domain=Reals, bounds=(None, None))
Master.intVars = Var(Master.intIndices, domain=NonNegativeIntegers,
                     bounds=(0, 10))
Master.dualVars = Var(Master.dualVarSet, domain=Reals, bounds=(None, None))


def objective_rule(model):
    return model.theta


Master.objective = Objective(rule=objective_rule, sense=maximize)


def theta_constraint_rule(model, k):
    return (model.theta <=
            sum(OBJ[j] * model.int_part_list[k][j] for j in INTVARS)
            - sum(OBJ[j] * model.intVars[j] for j in INTVARS)
            + sum(MAT[(i, j)] * model.dualVars[(i, k)] * (model.intVars[j] -
                                                          model.int_part_list[k][j])
                  for j in INTVARS for i in CONS))


Master.theta_constraint = Constraint(Master.intPartList,
                                     rule=theta_constraint_rule)


def dual_constraint_rule(model, j, k):
    return (sum(MAT[(i, j)] * model.dualVars[(i, k)] for i in CONS) <= OBJ[j])


Master.dual_constraint = Constraint(Master.conIndices, Master.intPartList,
                                    rule=dual_constraint_rule)

Master.int_part_list = [dict((i, 0) for i in INTVARS)]

debug_print = False
opt = SolverFactory("couenne")
for i in range(max_iters):
    listTemp.append(i)
    instance = Master.create_instance()
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    print('Solution in iteration', i)
    for j in instance.intVars:
        print(j, instance.intVars[j].value)
    print('Theta:', instance.theta.value)
    if instance.theta.value < .01:
        print("Finished!")
        for int_part in Master.int_part_list:
            print('Solution:', int_part)
            print('Right-hand side: [', )
            for k in CONS:
                print(sum(MAT[(k, l)] * int_part[l]
                          for l in INTVARS), )
            print(']')
        break
    if debug_print:
        for i in instance.dualVars:
            print(i, instance.dualVars[i].value)
        print(instance.dualVars[(0, 0)].value)
        for k in range(len(Master.int_part_list)):
            for j in CONVARS:
                print(j, k, OBJ[j], )
                print(sum(MAT[(i, j)] * instance.dualVars[(i, k)].value
                          for i in CONS))
        for k in range(len(Master.int_part_list)):
            for j in INTVARS:
                print(k, sum((MAT[(i, j)]) * instance.dualVars[(i, k)].value -
                             OBJ[j] for i in CONS), )
    Master.int_part_list.append(dict((i, round(instance.intVars[i].value))
                                     for i in INTVARS))



#policy
import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):

        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux



"""
import csv
from random import Random

from rlglue.agent import AgentLoader as AgentLoader
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3


class weak_baseline(Agent):
    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()
    obs_specs = []
    actions_specs = []
    policyFrozen=False
    exploringFrozen=False
    task=9

    u_err = 0   # forward velocity
    v_err = 1   # sideways velocity
    w_err = 2   # downward velocity
    x_err = 3   # forward error
    y_err = 4   # sideways error
    z_err = 5   # downward error
    p_err = 6   # angular rate around forward axis
    q_err = 7   # angular rate around sideways (to the right) axis
    r_err = 8   # angular rate around vertical (downward) axis
    qx_err = 9  # <-- quaternion entries, x,y,z,w   q = [ sin(theta/2) * axis; cos(theta/2)],
    qy_err = 10 # where axis = axis of rotation; theta is amount of rotation around that axis
    qz_err = 11

    def write_data(self,data,dtype):
        with open(dtype+"_"+str(self.task)+".csv",'a') as fp:
            writer = csv.writer(fp,delimiter=',')
            writer.writerow(data)

    def agent_init(self,taskSpecString):
        # print taskSpecString
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        if TaskSpec.valid:
            print len(TaskSpec.getDoubleActions()),": ",TaskSpec.getDoubleActions(),'\n',len(TaskSpec.getDoubleObservations()),": ",TaskSpec.getDoubleObservations()
            assert len(TaskSpec.getIntObservations())==0, "expecting no discrete observations"
            assert len(TaskSpec.getDoubleObservations())==12, "expecting 12-dimensional continuous observations"

            assert len(TaskSpec.getIntActions())==0, "expecting no discrete actions"
            assert len(TaskSpec.getDoubleActions())==4, "expecting 4-dimensional continuous actions"

            self.obs_specs = TaskSpec.getDoubleObservations()
            self.actions_specs = TaskSpec.getDoubleActions()
            # print "Observations: ",self.obs_specs
            # print "actions_specs:", self.actions_specs

        else:
            print "Task Spec could not be parsed: "+taskSpecString;

        self.lastAction=Action()
        self.lastObservation=Observation()

    def agent_policy(self, observation):
        o = observation.doubleArray
        weights = [0.0196, 0.7475, 0.0367, 0.0185, 0.7904, 0.0322, 0.1969, 0.0513, 0.1348, 0.02, 0, 0.23]

        y_w = 0
        roll_w = 1
        v_w = 2
        x_w = 3
        pitch_w = 4
        u_w = 5
        yaw_w = 6
        z_w = 7
        w_w = 8
        ail_trim = 9
        el_trim = 10
        coll_trim = 11

        # Collective Control
        coll = weights[z_w] * o[self.z_err] + weights[w_w] * o[self.w_err] + weights[coll_trim]

        # Forward-Backward Control
        elevator =  -weights[x_w] * o[self.x_err] - weights[u_w] * o[self.u_err] + weights[pitch_w] * o[self.qy_err] + weights[el_trim]

        # Left-Right Control
        aileron = -weights[y_w] * o[self.y_err] + -weights[v_w] * o[self.v_err] - weights[roll_w] * o[self.qx_err] + weights[ail_trim]

        rudder = -weights[yaw_w] * o[self.qz_err]

        action = [aileron, elevator, rudder, coll]

        self.write_data(action,"action")
        return action

    def agent_start(self,observation):
        print "Observation: ",observation.doubleArray
        returnAction = Action()
        returnAction.doubleArray = self.agent_policy(observation)

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        self.write_data(observation.doubleArray,"observation")
        return returnAction

    def agent_step(self,reward, observation):
        print "Observation: ",observation.doubleArray
        print "Reward: ",reward
        returnAction = Action()
        returnAction.doubleArray = self.agent_policy(observation)

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        self.write_data(observation.doubleArray,"observation")
        self.write_data([reward],"reward")
        return returnAction

    def agent_end(self,reward):
        print "Agent Down!"

    def agent_cleanup(self):
		pass

    def agent_message(self,inMessage):
        print inMessage

        return "Message received"

if __name__=="__main__":
	AgentLoader.loadAgent(weak_baseline())







import numpy as np

class LogisticPolicy:

    def __init__(self, θ, α, γ):
        # Initialize paramters θ, learning rate α and discount factor γ

        self.θ = θ
        self.α = α
        self.γ = γ

    def logistic(self, y):
        # definition of logistic function

        return 1/(1 + np.exp(-y))

    def probs(self, x):
        # returns probabilities of two actions

        y = x @ self.θ
        prob0 = self.logistic(y)

        return np.array([prob0, 1-prob0])

    def act(self, x):
        # sample an action in proportion to probabilities

        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)

        return action, probs[action]

    def grad_log_p(self, x):
        # calculate grad-log-probs

        y = x @ self.θ
        grad_log_p0 = x - x*self.logistic(y)
        grad_log_p1 = - x*self.logistic(y)

        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        # dot grads with future rewards for each action in episode

        return grad_log_p.T @ discounted_rewards

    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.γ + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob,action in zip(obs,actions)])

        assert grad_log_p.shape == (len(obs), 4)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # gradients times rewards
        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)

        # gradient ascent on parameters
        self.θ += self.α*dot


#policy evaluation using best reward


def run_episode(env, policy, render=False):

    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)



# ERRORED

class PolicyNetwork(ModelBase):
    def __init__(self,
                 preprocessor,
                 input_size=80*80,
                 hidden_size=200,
                 gamma=0.99):  # Reward discounting factor
        super(PolicyNetwork, self).__init__()
        self.preprocessor = preprocessor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.add_param('w1', (hidden_size, input_size))
        self.add_param('w2', (1, hidden_size))

    def forward(self, X):
        """Forward pass to obtain the action probabilities for each observation in `X`."""
        a = np.dot(self.params['w1'], X.T)
        h = np.maximum(0, a)
        logits = np.dot(h.T, self.params['w2'].T)
        p = 1.0 / (1.0 + np.exp(-logits))
        return p

    def choose_action(self, p):
        """Return an action `a` and corresponding label `y` using the probability float `p`."""
        a = 2 if numpy.random.uniform() < p else 3
        y = 1 if a == 2 else 0
        return a, y

"""

