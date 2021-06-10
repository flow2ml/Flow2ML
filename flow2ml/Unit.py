import numpy as np
import matplotlib.pyplot as plt

class Unit:
    
    def __init__(self,activation):
        ''' Sets the activation function as provided by the input.
            Args :
                activation: (String). Denotes the type of activation function to be used.
        '''
        self.activation=activation
        print("Activation set as {}".format(self.activation))
        
    def sigmoid(self,X):
        '''
            Applies the sigmoid activation function to the input.
            Args :
                X : (int or float). The input value on which the activation needs to be applied.
            Returns :
                sigmoid_value: applies sigmoid activation function on the input and returns the value.
        '''
        return 1/(1+np.exp(-X));
  
    def tanh(self,X):
        '''
            Applies the tanh activation function to the input.
            Args :
                X : (int or float) The input value on which the activation needs to be applied.
            Returns :
                tanh_value: applies tanh activation function on the input and returns the value.
        '''
        return np.tanh(X);

    def relu(self,X):
        '''
            Applies the relu activation function to the input.
            Args :
                X : (int or float) The input value on which the activation needs to be applied.
            Returns :
                relu_value: applies relu activation function on the input and returns the value.
        '''
        return np.maximum(0, X);
        
    def getActivation(self):
        '''
            Plots the activation function given as input.
        '''
        # plot the activation function using matplotlib library
        cur_axes = plt.gca()
        # to remove the x axis
        cur_axes.axes.get_xaxis().set_visible(False)
        rangex=np.linspace(-10, 10, 100)
        
        if(self.activation=="sigmoid"):
            plt.plot(rangex,self.sigmoid(rangex))
        elif(self.activation=="tanh"):
            plt.plot(rangex,self.tanh(rangex))
        elif(self.activation=="relu"):
            plt.plot(rangex,self.relu(rangex))
        plt.title("activation function = "+self.activation)
        
    def train(self,epochs,inputmatrix,outputmatrix):
        '''
            Trains the model.
            Args :
                epochs : (int). number of epochs the model needs to be trained. 
        '''

        # Initializing the weights randomly.
        self.weights=np.random.rand(inputmatrix.shape[1],1)

        self.trainingOutput=[]
        self.correctOutput=[]

        self.epochs=np.asmatrix(np.arange(0,epochs,1))

        # training over epochs given as input by the user.
        for i in range(0,epochs):

            # Getting the output from the unit.
            val=self.output(inputmatrix,outputmatrix,i)
            
            if(self.activation=="sigmoid"):
                val=val>=0.499
            elif(self.activation=="tanh" or self.activation=="relu"):
                val=val>=0
                    
            self.trainingOutput.append(val)
            self.correctOutput.append(outputmatrix)
            
        self.trainingOutput=np.asarray(self.trainingOutput)
        self.correctOutput=np.asarray(self.correctOutput)
            
    def output(self,inputmatrix,outputmatrix,iterno):
        '''
            1. computes the dot product of the weights,
            2. computes cost,
            3. optimizes the weights.
            Args :
                inputmatrix : The input value.
                outputmatrix : The truth values.
                iterno : the iteration number.
            Returns :
                self.result: the output after all computations.
        '''
        # Dot product of inputs and weights.
        self.ipv=np.dot(inputmatrix,self.weights)

        # Applying the activation function.
        if(self.activation=="sigmoid"):
            self.result=self.sigmoid(self.ipv)
        elif(self.activation=="tanh"):
            self.result=self.tanh(self.ipv)
        elif(self.activation=="relu"):
            self.result=self.relu(self.ipv)

        # Computing the cost.
        cost=self.costfunc(inputmatrix,outputmatrix)
        print("cost for {} iteration is {}".format(iterno,cost))

        # Updating the weights
        self.optimization(0.01,inputmatrix,outputmatrix)

        return self.result
        
    def costfunc(self,inputmatrix,outputmatrix):
        '''
            computes the cost.
            Args :
                inputmatrix : The input value.
                outputmatrix : The truth values.
            Returns :
                cost: The output of cost function.
        '''
        A1=np.multiply(outputmatrix,np.log(self.result))
        A2=np.multiply((1-outputmatrix),np.log((1-self.result)))
        cost=A1+A2
        cost=(-1/len(inputmatrix))*cost
        cost=np.sum(cost)
        return cost
    
    def optimization(self,learningRate,inputmatrix,outputmatrix):
        '''
            optimizes the weights.
            Args :
                learningRate : The learning rate to optimize the weights.
                inputmatrix : The input value.
                outputmatrix : The truth values.
        '''
        delta=(learningRate/len(self.weights))*np.dot(inputmatrix.transpose(),self.result-outputmatrix)
        self.weights=self.weights-delta
                                    
    def parameters(self):
        '''
            Returns the parameters of the logistic unit.
            Returns :
                parameters: (Dictionary). contains weights and activation function of the logistic unit.
        '''
        parameters={"weights":self.weights,"activation":self.activation}
        print("Weights = {}".format(parameters["weights"]))
        print("activation = {}".format(parameters["activation"]))
        return parameters
    
    def predict(self,inp):
        '''
            Predicts the output of logistic unit after training.
            Args :
                inp : The input to the logistic unit.
        '''
        # Dot product of inputs and trained weights.
        self.intake=np.dot(inp,self.weights)

        if(self.activation=="sigmoid"):
            self.out=self.sigmoid(self.intake)
        elif(self.activation=="tanh"):
            self.out=self.tanh(self.intake)
        elif(self.activation=="relu"):
            self.out=self.relu(self.intake)
        
        # returns the output.
        return self.out
    
    def results(self):
        '''
            Plots the training accuracy and training loss over each epoch.
        '''
        ls=self.correctOutput - self.trainingOutput

        loss=np.sum(ls,axis=1)
        loss=loss/len(loss)

        ac=np.sum(self.correctOutput==self.trainingOutput,axis=1)
        accuracy=ac/len(self.correctOutput)

        # Plotting the graph.
        plt.subplot(1,2,1)
        plt.plot(self.epochs.T,accuracy)
        plt.title("Training Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.subplot(1,2,2)
        plt.plot(self.epochs.T,loss)
        plt.title("Training Loss")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        #print("Accuracy : {}".format(accuracy[len(accuracy)-1]))
        #print("loss : {}".format(loss[len(loss)-1]))
        
    @classmethod
    def info(cls):
        print("This is a neural network unit which takes the output of previous unit and corresponding weights.\n It takes the product of both and is passes on to the activation function to get the output")