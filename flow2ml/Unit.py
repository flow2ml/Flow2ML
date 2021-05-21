import numpy as np
import matplotlib.pyplot as plt

class Unit:
    
    def __init__(self,activation):
        # self.activation="sigmoid"
        self.activation=activation
        print("Activation set as {}".format(self.activation))
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X));
  
    def tanh(self,X):
        return np.tanh(X);

    def relu(self,X):
        return np.maximum(0, X);
        
    def getActivation(self):
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
        self.weights=np.random.rand(inputmatrix.shape[1],1)
        self.trainingOutput=[]
        self.correctOutput=[]
        self.epochs=np.asmatrix(np.arange(0,epochs,1))
        for i in range(0,epochs):
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
        self.ipv=np.dot(inputmatrix,self.weights)
        if(self.activation=="sigmoid"):
            self.result=self.sigmoid(self.ipv)
        elif(self.activation=="tanh"):
            self.result=self.tanh(self.ipv)
        elif(self.activation=="relu"):
            self.result=self.relu(self.ipv)
        cost=self.costfunc(inputmatrix,outputmatrix)
        print("cost for {} iteration is {}".format(iterno,cost))
        self.optimization(0.01,inputmatrix,outputmatrix)
        return self.result
        
    def costfunc(self,inputmatrix,outputmatrix):
        A1=np.multiply(outputmatrix,np.log(self.result))
        A2=np.multiply((1-outputmatrix),np.log((1-self.result)))
        cost=A1+A2
        cost=(-1/len(inputmatrix))*cost
        cost=np.sum(cost)
        return cost
    
    def optimization(self,learningRate,inputmatrix,outputmatrix):
        delta=(learningRate/len(self.weights))*np.dot(inputmatrix.transpose(),self.result-outputmatrix)
        self.weights=self.weights-delta
                                    
    def parameters(self):
        parameters={"weights":self.weights,"activation":self.activation}
        print("Weights = {}".format(parameters["weights"]))
        print("activation = {}".format(parameters["activation"]))
        return parameters
    
    def predict(self,inp):
        self.intake=np.dot(inp,self.weights)
        if(self.activation=="sigmoid"):
            self.out=self.sigmoid(self.intake)
        elif(self.activation=="tanh"):
            self.out=self.tanh(self.intake)
        elif(self.activation=="relu"):
            self.out=self.relu(self.intake)
        return self.out
    
    def results(self):
        ls=self.correctOutput - self.trainingOutput
        loss=np.sum(ls,axis=1)
        loss=loss/len(loss)
        ac=np.sum(self.correctOutput==self.trainingOutput,axis=1)
        accuracy=ac/len(self.correctOutput)
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