
# From Deepmind paper

* Our new method uses a deep neural network f θ with parameters θ.
* In each position s, an MCTS search is executed, guided by the neural
network f θ .
* Neural network training in AlphaGo
Zero. The neural network takes the raw board position s t as its input, passes it through many convolutional layers
with parameters θ, and outputs both a vector p t , representing a probability distribution over moves, and a scalar value
v t , representing the probability of the current player winning in position s t . The neural network parameters θ are
updated so as to maximise the similarity of the policy vector p t to the search probabilities π t , and to minimise the
error between the predicted winner v t and the game winner z