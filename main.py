from collections import defaultdict
import numpy
import argparse


class RidgeRegressor:
    def __init__(self, A0, b0):
        (m, n) = A0.shape
        assert m == n
        self.m = m
        self.A = A0
        self.b = b0
        self._A_inv = None

    def update(self, x, y):
        assert x.shape == (self.m, )
        self.A += numpy.outer(x, x)
        self.b += x * y
        self._A_inv = None

    @property
    def A_inv(self):
        if self._A_inv is None:
            self._A_inv = numpy.linalg.inv(self.A)
        return self._A_inv

    def predict(self, x):
        theta = numpy.dot(self.A_inv, self.b)
        return numpy.dot(theta, x)

    def variance_multiplier(self, x):
        return numpy.dot(x, numpy.dot(self.A_inv, x)) ** 0.5

"""
protocol

("action" <-> "arm" (bandit lingo))

you are given a list of actions
you choose an action
you say which action you chose
you are informed of the reward
"""


class Scenario:
    def get_actions(self):
        """() -> [Action]"""
        pass

    def evaluate_action(self, action):
        """Action -> reward"""
        pass


class LinearScenario:
    def __init__(self, rewards, k, sparsity, scale):
        self.rewards = rewards
        self.k = k
        self.sparsity = sparsity
        self.scale = scale

    @property
    def _n(self):
        return len(self.rewards)

    def _make_action(self):
        x = numpy.random.uniform(0.0, 1.0, self._n) >= self.sparsity
        return x

    def get_actions(self):
        return [self._make_action() for _ in xrange(self.k)]

    def evaluate_action(self, action, scale=None):
        if scale is None:
            scale = self.scale
        if scale == 0.0:
            noise = 0
        else:
            noise = numpy.random.normal(0.0, scale=scale)
        return numpy.dot(action, self.rewards) + noise

class Policy:
    def decide(self, actions):
        """[Action] -> Action"""
        pass

    def learn(self, action, reward):
        """Action * reward -> ()"""
        pass

class CheatingPolicy:
    def __init__(self, scenario):
        self._scenario = scenario

    def decide(self, actions):
        # cheat outrageously
        expected_reward = lambda a : self._scenario.evaluate_action(a,
                scale=0.0)
        return max(actions, key=expected_reward)

    def learn(self, action, reward):
        pass

class EpsilonGreedyPolicy:
    def __init__(self, epsilon, regressor):
        self._epsilon = epsilon
        self._regressor = regressor

    def decide(self, actions):
        explore = numpy.random.uniform(0.0, 1.0) <= self._epsilon

        if explore:
            i = numpy.random.randint(len(actions))
            return actions[i]
        else:
            return max(actions, key=self._regressor.predict)

    def learn(self, action, reward):
        self._regressor.update(action, reward)

class UcbPolicy:
    def __init__(self, regressor, delta=None, alpha=None, collect_diagnostics=False):
        self._delta = delta
        if alpha is None:
            alpha = make_alpha(delta)
        self._alpha = alpha
        self._regressor = regressor
        self._collect_diagnostics=collect_diagnostics
        self._history = []

    def _ucb(self, action):
        return (self._regressor.predict(action) + self._alpha *
            self._regressor.variance_multiplier(action))

    def decide(self, actions):
        return max(actions, key=self._ucb)

    def learn(self, action, reward):
        if self._collect_diagnostics:
            bound = self._ucb(action)
            self._history.append((action, bound, reward))
        self._regressor.update(action, reward)

    def diagnostics(self):
        print 'UcbPolicy diagnostics:'
        print '\talpha = %r' % self._alpha
        print '\tdelta = %r' % self._delta

        _, bounds, rewards = zip(*self._history)
        frac_reward_leq_bound = numpy.mean(numpy.asarray(rewards) <=
                numpy.asarray(bounds))
        print '\tfrac reward leq bound = %r' % frac_reward_leq_bound



def make_alpha(delta):
    """0 < delta < 1 : probability that you want the upper confidence bound to
    hold. i dont understand the derivation of this"""
    assert delta > 0.0
    return 1.0 + (numpy.log(2.0/delta)/2.0) ** 0.5

def main(n, trials, dbg):
    rewards = numpy.random.normal(0.0, 1.0, n)
    k = 10 # number of actions per event
    sparsity = 0.8 # fraction of feature components nonzero
    scale = 5.0 # scale of gaussian noise added to rewards

    scenario = LinearScenario(rewards, k, sparsity, scale)

    def make_rr():
        A0 = numpy.eye(n)
        b0 = numpy.zeros(n)
        return RidgeRegressor(A0, b0)

    policies = {
        'cheating' : CheatingPolicy(scenario),
        'greedy_0.0' : EpsilonGreedyPolicy(0.0, make_rr()),
        'greedy_0.1' : EpsilonGreedyPolicy(0.1, make_rr()),
        'greedy_1.0' : EpsilonGreedyPolicy(1.0, make_rr()),
        'ucb_0.500' : UcbPolicy(make_rr(), delta=0.500, collect_diagnostics=dbg),
        'ucb_0.100' : UcbPolicy(make_rr(), delta=0.100, collect_diagnostics=dbg),
        'ucb_0.010' : UcbPolicy(make_rr(), delta=0.010, collect_diagnostics=dbg),
        'ucb_0.001' : UcbPolicy(make_rr(), delta=0.001, collect_diagnostics=dbg),
    }

    history = defaultdict(list)

    def display_progress(title):
        print '=== %s ===' % title
        for pname in sorted(history):
            net_reward = numpy.sum(history[pname])
            print '%r\t%r' % (pname, net_reward)

    for t in xrange(trials):
        actions = scenario.get_actions()
        # n.b. since evaluation of actions may be non-deterministic,
        # do it once here to de-randomise it across policies
        rewards = {tuple(a):scenario.evaluate_action(a) for a in actions}
        for pname in sorted(policies):
            p = policies[pname]
            a = p.decide(actions)
            reward = rewards[tuple(a)]
            p.learn(a, reward)
            history[pname].append(reward)

        display_progress('scores after %r trials' % t)

    if dbg:
        print '=== diagnostics ==='
        for pname in sorted(policies):
            p = policies[pname]
            if not hasattr(p, 'diagnostics'):
                continue
            print 'policy %r' % pname
            p.diagnostics()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', action='store_true', default=False)
    p.add_argument('--debug', action='store_true', default=False)
    p.add_argument('--trials', type=int, default=1000)
    p.add_argument('-n', type=int, default=100, help='feature dimension')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()

    def go():
        main(trials=args.trials, dbg=args.debug, n=args.n)

    if args.profile:
        import cProfile as profile
        import pstats

        p = profile.Profile()
        p.runcall(go)
        s = pstats.Stats(p)
        s.sort_stats('cumulative').print_stats(20)
    else:
        go()

