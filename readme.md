### Note 1: The following repository grew out of [Romain Brette's critical thought experiment on the free energy principle](http://romainbrette.fr/what-is-computational-neuroscience-xxix-the-free-energy-principle/). 
### Note 2: This Github repository is accompanied by [the following blog post](http://paulispace.com/statistics/2018/04/07/causal_path_entropy.html). 

| ![image-1.jpg](https://raw.githubusercontent.com/pauli-space/Free_Energy_experiments/master/diagram.png) | 
|:--:| 
| *diagram of the system* |

To simulate Romain's problem, I made the following assumptions:

1. We have an organism which has to eat at least $$k$$ times in the last 24 hours and can eat at most once per hour. 
2. The homeostatic conditions of our organism are given by a Gaussian distribution centered at $$k$$ with unit variance, a Gaussian food critic if you will. This specifies that our organism should't eat much less than $$k$$ times a day and shouldn't eat a lot more than $$k$$ times a day. In fact, this explains why living organisms tend to 
have masses that are normally distributed during adulthood. 
3. A food policy consists of a 24-dimensional vector where the values range from 0.0 to 1.0 and we want to maximise the negative log probability that the total consumption is drawn from the Gaussian food critic. 
4. Food policies are the output of a generative neural network(setup using TensorFlow) whose inputs are either one or zero to indicate a survival prior, with one indicating a preference for survival. 
5. The backpropagation algorithm, in this case Adagrad [5], functions as a homeostatic regulator by updating the network with variations in the network weights proportional to the negative logarithmic loss(i.e. surprisal). 

Assuming $$k=3$$, I ran a simulation in the [following notebook](https://github.com/pauli-space/Free_Energy_experiments/blob/master/simulation.ipynb) and found that the discovered food policy differs significantly from Romain's expectation that the agent would choose to not look for food in order to minimise surprisal. In fact, our simple agent manages to get three meals per day on average so it survives. 


# Motivation:

1. Until recently, the Free Energy Principle has been a constant source of mockery from neuroscientists who misunderstood it and so I hope that by growing a collection 
of free-energy motivated reinforcement learning examples on Github we may finally have a constructive discussion between scientists
2. The details of the first experiment are contained in the [following notebook](https://github.com/pauli-space/Free_Energy_experiments/blob/master/simulation.ipynb). 


## References:

1. The free-energy principle: a rough guide to the brain? (K. Friston. 2009.)
2. The Markov blankets of life: autonomy, active inference and the free energy principle (M. Kirchhoff, T. Parr, E. Palacios, K. Friston and J. Kiverstein. 2018.)
3. Free-Energy Minimization and the Dark-Room Problem (K. Friston, C. Thornton and A. Clark. 2012.)
4. What is computational neuroscience? (XXIX) The free energy principle (R. Brette. 2018.)
