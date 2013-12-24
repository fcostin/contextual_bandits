mucking about with simple contextual bandit algorithms
======================================================

references
----------

1.  Li, Chu, Langford, Schapire - A contextual bandit approach to personalized news article recommendation [2010]
2.  Chu, Li, Reyzin, Schapire - Contextual Bandits with linear payoff functions [2011]


### warning about the above references:

the above approaches make the assumption that the expected reward
function is actually a linear function of the features.

in practice this is unlikely to be the case.

it would be good to know how the approach holds up / degrades if
the expected reward function is only "approximately" linear,
or perhaps not at all linear...


linear least squares
--------------------

(with a ridge regression term, too...)

    you get to see (x, y) pairs

    you gotta estimate theta s.t.

    theta^t x = y

    min_{theta} || X theta - y ||^2 + ||theta||^2

    ( X^T X + I ) theta = X^T y

    --------------------------

    X : (n_samples, n_features)


    A   :=  X^T X + I
    b   :=  X^T y

    A : (n_features, n_features)
    b : (n_features, )

    A_0 = I
    b_0 = 0

    updating it:
        x : new feature vector, shape (n_features, 1)
        y : new response, scalar

    A <- Aprev + xx^T
    b <- bprev + xy

