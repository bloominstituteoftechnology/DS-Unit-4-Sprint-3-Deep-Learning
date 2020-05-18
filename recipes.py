#! /opt/conda/envs/intro_nn/bin/python
# rnn and lstm
# looking at how the models work at a high level.
# using ltsm and rnn to generate text after processing it.
# sequence based modeling, similar to time seris, basicly where the order of anylisi matters
# python list's are a good example of squence
# nural networks are not great at predicting stock information,
# they are really good at modeling more complex data such as sequence's of texts
# the order of text data is the order that we are reading the information
# thats the modivation of using nn as models.

# recursion is the relation in a mathmatical equation to define a recurssion, when the value of the
# current squence is one number ago added to 2 number's ago.
# in our case we will have a nn unit then we are going to loop over it multiple times
# to create an output. this is rnn a nn that loops over it'self.
# one way to denote this is by using a folded nn diagrame.
# the model has to start somewhere but that we have to pretend that there is information coming from before the
# start point in the recursion
# the predicted output can be the input to the next step in the loop.
# one of the most effective way to structure that loopis by using lstm this enables
# remebering and forgetting. in rnn there is a problem learning the long term relationships

# menthioning a character in the first chaper but not referencing that character until chap9
# lstm parameterizes the remebring that character.
# lstm will still prioritize things in stm but the information for 10000 characters ago will not be dead node
# using attention will focus on key peices of information like lstm.

# rnn have the vanishing gradient problems, lstm is better at this problems but not perfect
# vanishing gradients mean that the updates to the weights are soo small that the can effectivly be 0
# rnn and lstm's are good in calssification problems.

#read the reasonable effectiveness of neural networks blog post.
# the soffical tensorflow doc on rnn models references that blog post.
# most of the work that is going to be done with lstm's are going to be text classification
# it's not going to be used to generate text, we should use it for generations because it's on
# of the fundemental's that is included in the models tipe


class lstm():
    def __init__(self):
        return None

    def fit(self, x_train, y_train, x_val, y_val):

        return None

    def predict(self, X):
        return None

    def build_model(self, params):
        model = Sequental()
        return model
