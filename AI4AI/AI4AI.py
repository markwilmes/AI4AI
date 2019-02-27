# intelligent load balancing for training
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from random import randint

#mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

class GenerateNetwork: # methods to create a neural network
    def __init__(self):
        self.network = {}
        self.accuracy = 0
        self.compile_methods = {}
        self.early_stopping = k.callbacks.EarlyStopping(patience=50)

    def generate_raw_network(self,nn_params):       
        for i in nn_params:
            self.network[i] = np.random.choice(nn_params[i])
        return self.network

    def generate_random_compile(self,compile_params):
        for i in compile_params:
            self.compile_methods[i] = np.random.choice(compile_params[i])
        return self.compile_methods

    def set_net_params(self,network):
        for i in network:
            self.network[i] = network[i]

    def set_compile_params(self,compiles):
        for i in compiles:
            self.compile_methods[i] = compiles[i]

    def create_net(self,network,compile_params):
        self.network = network
        self.compile_methods = compile_params
        return self

    def compile_net(self,input_shape,classes):
        layers = self.network['layers']
        neurons = self.network['neurons']
        activation = self.network['activation']
        outActivation = self.network['activation_out']
        net = k.Sequential()

        for i in range(layers):
            if(i == 0):
                net.add(k.layers.Dense(neurons,activation=activation,input_shape=input_shape))
            else:
                net.add(k.layers.Dense(neurons,activation=activation))
        net.add(k.layers.Dense(classes,activation=outActivation))

        loss = self.compile_methods['loss']
        optimizer = self.compile_methods['optimizer']
        metrics = self.compile_methods['metrics']

        net.compile(loss=loss,optimizer=optimizer,metrics=[metrics])
        return net

    def train(self,data): # all dictionaries here
        input_shape = data['input_shape']
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        classes = data['num_classes']
        
        #set_net_params(network)
        #set_compile_params(compile_params)

        model = self.compile_net(input_shape,classes)
        model.fit(x_train,y_train,epochs=10000,verbose=0,validation_data=(x_test,y_test),callbacks=[self.early_stopping])

        score = model.evaluate(x_test,y_test,verbose=0)
        self.accuracy = score[1]


class GeneticOptimization: # methods to perform genetic optimization on the nns
    def __init__(self, nn_params, compile_params, retain=0.3, mutate_chance=0.1): # initialize the Genetic alorithms object
        self.mutate_chance = mutate_chance
        #self.random_select = random_select
        self.retain = retain
        self.nn_params = nn_params
        self.compile_params = compile_params

    def create_population(self,num_nets):
        nets = []
        for i in range(num_nets):
            network = GenerateNetwork()
            nets.append(network.create_net(network.generate_raw_network(self.nn_params),network.generate_random_compile(self.compile_params)))
        return nets
    
    def fitness_test(self,nets):
        chance_of_selection = []
        for i in nets:
            chance_of_selection.append(i.accuracy)
        #chance_of_selection.sort(reverse=True)
        #for i in range(len(chance_of_selection)):
        #    chance_of_selection[i] = chance_of_selection[i]*(len(nets)-i)
        #division = sum(chance_of_selection)
        #for i in range(len(chance_of_selection)): # either normilize probs to 1, or find new way to select items    
        #    chance_of_selection[i] = chance_of_selection[i]/division # selection is too random, does not converge on better population
        new_nets = {}
        for i in range(len(chance_of_selection)):
            new_nets[nets[i]] = chance_of_selection[i]
        #new_nets.keys()).sort(reverse=True)
        #mother,father = np.random.choice(nets,2,p=chance_of_selection,replace=False)
        counter = 0
        end = len(new_nets)*self.retain
        parents = []
        for i in sorted(new_nets, key=new_nets.__getitem__, reverse=True):
            if(counter < end):
                parents.append(i)
            else:
                break
            counter += 1
        return parents

    def get_parents(self,new_nets):
        mother,father = np.random.choice(new_nets,2,replace=False)
        return mother,father


    def spawn_net(self,mother,father):
        children = []
        for i in range(2):
            child = {'nn':{},'compile':{}}
            for param in self.nn_params:
                child['nn'][param] = np.random.choice([mother.network[param], father.network[param]])
            for param in self.compile_params:
                child['compile'][param] = np.random.choice([mother.compile_methods[param],father.compile_methods[param]])

            network = GenerateNetwork()
            network.create_net(child['nn'],child['compile'])

            # Randomly mutate some of the children.
            if self.mutate_chance > np.random.random():
                network = self.mutate(network)

            children.append(network)
        return children

    #def kill_weak(self,nets):
    #    if(nets)


    def mutate(self, network):
        choice = np.random.choice(['compile','network'])
        if(choice == 'compile'): # mutate either compile or network but not both
            mutation = np.random.choice(list(self.compile_params.keys()))
            network.compile_methods[mutation] = np.random.choice(self.compile_params[mutation])
        else:
            mutation = np.random.choice(list(self.nn_params.keys()))
            network.network[mutation] = np.random.choice(self.nn_params[mutation])

        return network

    def runEvolution(self,data,generations,population_size):
        population = []
        nets = self.create_population(population_size) # initial populations
        population = nets
        generation_counter = 0
        while(generation_counter < generations): # loop through all generations
            print("Training generation {}".format(generation_counter))
            acc = 0
            accuracy = []
            for i in population: # train all nets to get accuracy measures
                i.train(data)
                accuracy.append(i.accuracy)
                acc += i.accuracy
            print("Average accuracy : {}".format(acc/generations))
            with open("average.txt",'a+') as f:
                f.write(str(acc/generations) + "\n")
            accuracy.sort(reverse=True)
            with open("top3.txt",'a+') as f:
                f.write(str(accuracy[0]) + "," + str(accuracy[1]) + "," + str(accuracy[2]) + "\n")
            print("Top 3 nets: {} {} {}".format(accuracy[0],accuracy[1],accuracy[2]))
            new_population = []
            if(generation_counter != generations - 1):
                print("Testing and building new generations")
                parents = self.fitness_test(nets)
                new_population.extend(parents)
                while(len(new_population) < len(population)): # create a new population
                    mother,father = self.get_parents(nets) # runs the fitness test function on each and selects best nets
                    # function below also runs mutation on children
                    children = self.spawn_net(mother,father) # crossover function that creates net from attributes of parents
                    for child in children:
                        if(len(new_population) < len(population)):
                            new_population.append(child)
                        else:
                            break
            generation_counter += 1
                
                
        
        #self.evolve(children) # crossover
        #self.mutate(children[0]) # mutation
        #pass
        # create initial nets
        # while not one, evolve nets
        

class IngestData:
    def __init__(self):
        pass

class FeatureExtraction:
    def __init(self):
        pass


def main(): 

    network_params = { # possible parameters to train on
        'neurons': [16, 32, 64, 128, 256, 512],
        'layers': [1, 2, 3, 4, 5, 6],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'activation_out': ['relu','elu','tanh','sigmoid','softmax']        
    }

    compile_params = { # possible compile args to compile on
        'loss':['mean_squared_error','mean_absolute_error','categorical_crossentropy','binary_crossentropy','cosine_proximity'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],
        'metrics': ['binary_accuracy','categorical_accuracy','accuracy']
    }

    '''
    input_shape = data['input_shape']
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        classes = data['num_classes']
    '''
    x = np.array([[5.0, -0.25, 0, 1.25, 0, -0.75, 2.125], [0.0, 1.5, 0, 1.5, 0, 0, -1.125], [0, 0, 1.75, 0.25, 0, 0, 0.0], [0, 0.375, 0.75, 0, 0, 1.5, -0.25], [0, 0.5, 0.0, 0.0, -0.375, -0.125, 0.625], [0, 0.0, 0, 0, 0, -0.375, 0.0], [0, 0.0, 0, 0.0, 0, 0, 0.25], [0.0, 0.5, 0, 0, 0, -0.875, 2.0], [0, 2.125, 0.625, 0, 0.5, 0.25, 0.25], [0, 0, 0.625, 1.625, 0, 0.5, 0.0], [1.5, 0, 1.125, 0, 0, -0.5, -0.25], [0, 0, -0.125, 0.0, 0, 1.625, -0.5], [0, 0, -0.5, 0, -0.625, 0, 0.0], [0, 0.875, 1.625, 0, 0, 0, -1.25], [-0.625, 0, 0.0, 0.375, 0, 0, 1.375], [0, 0, 0, 1.25, 1.375, 0, 2.5], [0, 0, 1.25, 0, 1.125, 0, 1.375], [0, 0.0, 0, 0.25, 0, 0, 0.375], [0, 0, 0, -0.125, 0.375, 0.0, 0.0], [0, 0, 0, 0, 0, 0, -0.375], [0.0, 0, 0.0, 0, 0, 0.0, 0.125], [1.0, 0, 0, 0.75, 0.125, 0.875, 0.875], [0, 0.125, 1.75, 0.0, 0, 0, 0.75], [0.75, 0.625, 0, -1.25, 0, 0, 1.5], [1.25, 0, 0, 0, 0, 0, 1.375], [0, 0.375, 0, 0, 0, -0.25, 0.375], [0, 0, 0.25, 2.125, 1.0, 0.25, -0.625], [0.0, 0.0, 0.0, 0, -0.375, -0.25, 0.875], [-0.25, 0, 0, 0.0, 0, 1.25, -0.25], [0.375, -0.375, 0, 0, 2.625, 0.0, 0.875], [0, 0.0, 0.0, 1.375, 0.5, 2.25, 2.875], [3.25, 0, 0.0, 0, 0, 0, -0.25], [0, 1.0, 0.0, 0.625, 1.75, 0.75, 0.25], [0, 0.125, 2.0, 1.0, 4.125, 1.875, 1.5], [-0.125, 1.25, 0, 0, 3.375, 2.25, 0.5], [1.25, 0.875, 0.375, 0, 0.25, 2.0, -4.375], [0, -1.875, 0, 0.125, 1.5, 0, -0.25], [0.875, 0.0, -0.25, -2.0, 0, 0, 0.0], [0, 0, 0, 0.0, 0, 0, -0.375], [0.0, 0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 0, 0.125], [0, -1.0, 0, 0.875, 0, 0, -0.25], [0, 2.625, 0, 0, 0, 0, 0.0]])
    y = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 1, 1], [0, 1, 0, 0, 1], [1, 1, 1, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 1], [0, 1, 0, 0, 1], [0, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 1], [1, 0, 0, 1, 1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 1, 1, 0], [0, 1, 0, 1, 1], [1, 0, 0, 0, 1]])
    xTest = np.array([[0.625, 0, 0.375, 0, 0, 0, 2.375], [-0.375, 0, 0.0, 0, 0, 0.0, -0.375], [0.75, 0, -0.5, -1.125, -0.5, 1.75, 0.0], [-0.875, 0.0, 0.0, 0, 0, 0, 0.5], [0, 0, 0, 0, 0, 0, 1.25], [-0.125, 1.375, 0.0, 0, 0.5, 0, 0.875], [0.625, 1.125, 0.25, 0.25, 0, 0.375, -0.5], [0.25, -0.5, 0.125, 0.375, -0.75, 0, 0.25], [0, 0, 0, 0, 0, 0.375, 0.0], [0, 0, -0.25, 1.75, -0.5, 1.5, 1.125], [5.5, -0.25, 0.0, 3.125, 0.25, 0.25, 2.0], [0.0, 0, 0, 0.625, 0.375, 0, 1.125], [0.0, 0, -0.75, 1.0, 0, 0.125, 4.375], [0.625, 0, -0.25, 1.375, 2.0, 0, 0.25], [0, 1.0, 0.875, 1.75, 0.625, -1.0, 0.0], [0.375, 0, 0, 1.625, -0.25, 0.0, 0.625], [0.375, 0, 0.0, 2.125, 0.875, 1.125, 1.5], [0.0, 0.625, -0.25, 0.25, -1.625, 0.0, -2.125], [0, -2.375, 0, 0.0, 0, 0, -0.5], [0, 0, 0, -0.25, 2.0, -1.125, 0.5], [0.625, 1.0, 0.25, -0.25, 0.375, 0.125, -0.5], [1.0, 3.5, 2.875, 0.75, 0.25, 1.625, 0.0], [0, 0.625, 0.875, 1.0, 0.0, 1.75, 0.25], [0.375, 0, 0.75, 0.25, 1.625, 0, 0.25], [-0.25, 0.0, -1.25, 0, 0, 0, 0.25], [0, 0, 0, 0, 1.375, 0.125, -1.0], [-1.75, 1.375, 0.0, 0.25, 0, 1.0, 1.625], [1.5, 0.25, 2.75, 0.375, 0, 0.625, 0.25], [-0.75, 0.5, 2.375, 0.5, 0, 2.0, -0.375], [0, 0.0, 0.0, 0.375, -0.5, 2.875, 0.0], [1.875, 0, 0.25, 0, 0, 0, 0.0], [2.125, 1.0, 0.0, 0.25, 0.25, 0, 0.375], [-2.125, 1.5, -0.625, 0.25, 0, 0.875, 1.5], [0, 0, 0, 0.0, 0, 0.625, 0.75], [-0.875, 0.375, 0, 0, 0.0, -0.375, 0.75], [0, 0, 0, 0, 0.25, 0.0, 0.0], [0, -0.375, 0.375, 3.875, 0.75, -0.25, 0.0], [0.0, 1.75, 0, 1.125, 0.375, 0.875, -1.0], [2.25, -0.125, 1.25, 1.375, 0.25, 1.5, 0.0], [0.0, 1.375, 3.875, 0, 0.5, 2.375, 0.0], [0, 0, -0.625, 1.5, 2.0, 2.875, 0.625], [-1.25, 0, 0.375, 0, 0, 0, 0.0], [0.0, 0.375, 0, -0.375, 0, 0, 0.0]])
    yTest = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [1, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 1, 0, 0, 1], [1, 1, 1, 0, 1], [0, 1, 0, 0, 1]])
    data = { # input data
        'x_train':x,
        'input_shape':(7,),
        'y_train':y,
        'num_classes':5,
        'y_test':yTest,
        'x_test':xTest
    }
    evolve = GeneticOptimization(network_params,compile_params) # create object
    evolve.runEvolution(data,10,20) # run evolution on object


    
    #evolve.create_population
    #pass
    
    # read through file, see if it can be parsed as json, then csv, then other filetypes
    # use pandas read_csv to do this, turn it into a dataframe
    # for each column in the dataframe create a tensorflow network based on the other columns in the dataframe (inputs = other columns, output = this colum)

if __name__ == '__main__':
    main()
