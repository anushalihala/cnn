
import numpy as np
import pdb
  

class Convolution:
    # 2D Convolution
    def __init__(self, filter_parameters, in_stride, in_padding):
        #INPUTS
        #filter_parameters: [number of filters, filter height or width]
        #in_stride: stride
        #in_stride: padding 
        
        self.filter_num = filter_parameters[0]
        self.filter_size = filter_parameters[1]
        self.filters = np.random.rand(filter_parameters[0],filter_parameters[1],filter_parameters[1])
        #self.bias=  np.zeros(self.filter_num) #TODO
        
        self.stride=in_stride
        self.padding=in_padding     
        
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height and width w
        #OUTPUT
        #numpy array of dimensions [m,f,o,o]; resulting layers after applying filters
        
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        #checking for single sample input
        if(len(input_data.shape)==2):
            input_data=np.array([input_data])  
            
        m=input_data.shape[0]
        w=input_data.shape[1]
        
        if(w<self.filter_size):
            return -1 #ERROR
        
        o = int(((w - self.filter_size + 2*self.padding)/self.stride) + 1) #output height or width
        output = np.zeros([m,self.filter_num,o,o])                         #initialising output with zeros
        
        #Adding padding
        input_data=np.pad(input_data,[(0,0),(self.padding,self.padding),(self.padding,self.padding)],'constant')
        
        for k in range(m):                          #for each input sample
            for f in range(self.filter_num):        #for each filter
                for i in range(o): 
                    i_start_index=i*self.stride
                    i_end_index=i_start_index+self.filter_size
                    
                    for j in range(o):  
                        j_start_index=j*self.stride
                        j_end_index=j_start_index+self.filter_size
                       
                        output[k,f,i,j] = np.sum(input_data[k, i_start_index:i_end_index ,  j_start_index:j_end_index] * self.filters[f]) #+ self.bias[f]
                       
        return output
    
    def backward_pass(self, input_data, output_error,learning_rate):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], of m samples of height,width = w
        #output_error: numpy array of dimensions [m,f,o,o] of errors
        #FUNCTION UPDATES WEIGHTS ACCORDINGLY
        
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        #checking for single sample input
        if(len(input_data.shape)==2):
            input_data=np.array([input_data])  
            
        m=input_data.shape[0]
        w=input_data.shape[1]
        o=output_error.shape[3]
        
        if(w<self.filter_size):
            return -1 #ERROR
        
        #Adding padding
        input_data=np.pad(input_data,[(0,0),(self.padding,self.padding),(self.padding,self.padding)],'constant')
        
        Delta_filters=np.zeros(self.filters.shape)
        
        for k in range(m):                          #for each input sample
            for f in range(self.filter_num):        #for each filter
                for i in range(o): 
                    i_start_index=i*self.stride
                    i_end_index=i_start_index+self.filter_size
                    
                    for j in range(o):  
                        j_start_index=j*self.stride
                        j_end_index=j_start_index+self.filter_size
                                               
                        Delta_filters[f]= Delta_filters[f] + input_data[k, i_start_index:i_end_index ,  j_start_index:j_end_index]*output_error[k,f,i,j]
                        
        Delta_filters=(1/m)*Delta_filters
        self.filters=self.filters-learning_rate*Delta_filters

class ReLU:
    # ReLU layer applies function f(x)=max(0,x) to input
    def __init__(self):
        pass
        
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,d,w,w], for m samples of depth = d, and height,width = w
        #OUTPUT
        #result of applying f(x) to all input values
        vmax=np.vectorize(max)
        return vmax(input_data,0)
    
    def backward_pass(self,input_data,output_error):
        #INPUTS
        #input_data: numpy array of dimensions [m,d,w,w], of m samples of depth = d, and height,width = w
        #output_error: numpy array of dimensions [m,d,w,w] of errors in output
        #OUTPUT
        #[m,d,w,w] array of errors in input layer
        
        output_error[input_data<0]=0
        return output_error
        
class FC:
    # Fully connected neural network with no hidden layer
    def __init__(self,i,j):
        # INPUTS
        # i=number of rows/number of neurons in output layer
        # j=number of columns/number of neurons in input layer
        
        self.W=np.random.rand(i,j+1) # Weights of network)
        
    def process_input(self,input_data):
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        bias = np.ones(1)
        input_vectors=[]
        for sample in input_data:
            input_vectors.append(np.append(bias,sample.ravel()))
        
        input_vectors=np.array(input_vectors)
        
        return input_vectors
    
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height,width = w
        #OUTPUT
        #values of output layer
        
        input_vectors=self.process_input(input_data)
        
        output= np.dot(self.W,input_vectors.T)
        
        output=self.sigmoid(output)
        
        return output
        
    def backward_pass(self, input_data, output_error, learning_rate):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height,width = w
        #output_error: numpy array of dimensions [m,i], (output-label) for i neurons in m samples
        #learning_rate: learning rate for weight updation
        #OUTPUT
        #[m,j] array of errors in input layer for each sample
        #FUNCTION UPDATES WEIGHTS ACCORDINGLY
        
        input_vectors=self.process_input(input_data)
        m=len(input_vectors)

        input_vectors=np.apply_along_axis(self.sigmoid_gradient,0,input_vectors)
        
        input_error = np.dot(output_error,self.W)*input_vectors
        
        Delta=np.zeros(self.W.shape)
        for i in range(m):
            temp=np.dot(output_error[i:i+1,:].T,input_vectors[i:i+1,:])
            Delta = Delta + temp
            
        Gradient=(1/m)*Delta
        
        self.W=self.W-learning_rate*Gradient
        
        return input_error[:,1::]
        
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    def sigmoid_gradient(self,x):
        sigmoid_val=self.sigmoid(x)
        return sigmoid_val*(1-sigmoid_val)
        
class CNN1:
    def __init__(self):
        #CNN with the following architecture:
        #Input - 32x32 matrix
        #Convolution layer with 4 filters ... Filter size = (3x3), Output of layer = (16x16x4)
        #ReLU layer
        #Fully connected layer with no hidden layers, with 1024 input neurons and 10 output neurons
        #Output  - 10x1 vector
        
        self.conv1=Convolution([4,3],2,1)
        self.relu1=ReLU()
        self.fc1=FC(10,1024)
        
    def compute_output(self, input_samples):
        #INPUTS
        #input_samples: array of dimensions [m,w x w], where there are m samples of height and width w
        
        #converting to dimensions [m,w,w]
        input_samples=self.augment_X(input_samples,0)
        
        # calculating activations for each layer
        a_conv = self.conv1.forward_pass(input_samples)
        
        #if forward pass computation of convolution layer unsuccessful
        if(isinstance(a_conv,int)):
            return
        
        a_relu=self.relu1.forward_pass(a_conv)
        a_fc=self.fc1.forward_pass(a_relu)
        
        output=a_fc.argmax(0)
        output=output+1
        
        # TESTING
        # print(a_conv)
        # print(a_relu)
        # print(a_fc)
        # print(output)
        
        return output
        
    def cost(self, output, labels):
        #INPUTS
        #output: numpy array of dimensions [m, 10] for m samples; values output by CNN
        #labels: numpy array of dimensions [m, 10] for m samples; labels for training data
        
        #ensuring inputs are numpy arrays
        
        if(not isinstance(output,np.ndarray)):   
            output=np.array(output)
        if(not isinstance(labels,np.ndarray)):   
            labels=np.array(labels)
          
        #checking for single sample values
        if(len(output.shape)==2):
            m=len(output)
        else:
            m=1

        return (1/m)*np.sum(-labels*np.log(output) - (1-labels)*np.log(1-output))
    
    def train(self, training_data):
        #INPUTS
        #training_data: numpy array of dimensions [m,1025] where there are m samples and last column is the corresponding label
        
        #ensuring input is numpy array
        if(not isinstance(training_data,np.ndarray)):   
            training_data=np.array(training_data)
         
        m=training_data.shape[0]
        k=training_data.shape[1]        
        
        X=training_data[:,0:k-1]
        pre_y=training_data[:,k-1]
           
        y=np.apply_along_axis(self.augment_label,0,pre_y)
        
        
        #converting to dimensions [m,w,w]
        X=self.augment_X(X,0)
        
        for count in range(200):
            # FORWARD PASSES
            a_conv = self.conv1.forward_pass(X) 
            #if forward pass computation of convolution layer unsuccessful
            if(isinstance(a_conv,int)):
                return
            a_relu=self.relu1.forward_pass(a_conv)
            a_fc=self.fc1.forward_pass(a_relu)
                
            #BACKWARD PASSES    
            error = a_fc.T-y
            #(TESTING)
            print(np.sum(error*error))  #ISSUE: sigmoid function in FC layer results in approximation which causes log0 calculation in self.cost
         
            error_relu=self.fc1.backward_pass(a_relu,error,0.5)
            # converting to dimensions [m,d,w,w]
            error_relu= self.augment_X(error_relu,1) 
            
            error_conv=self.relu1.backward_pass(a_conv,error_relu)
            
            self.conv1.backward_pass(X,error_conv,1)
            
        print("labels", y)
        print("CNN output",a_fc.T)
        
    def augment_label(self, simple_labels):
        #INPUTS
        #simple_labels: array of dimensions [m]; values range from 1 to 10; label x indicates output neuron x = 1 and all other output neurons = 0
        #OUTPUT
        #array of dimensions [m, 10] where for each i=1,2..m, [i,x]=1 and [i,not x]=0
        
        m=len(simple_labels)
        augmented_labels=np.zeros([m,10])
        for i in range(m):
            ith_label=simple_labels[i]
            augmented_labels[i,ith_label-1]=1
            
        return augmented_labels
        
    def augment_X(self, X, parameter):
        #INPUTS
        #X: numpy array of dimensions [m,1024]
        #parameter: parameter to choose which reshape takes place
        #OUTPUTS
        #numpy array of dimensions [m,32,32]
        
        #ensuring input is numpy array
        if(not isinstance(X,np.ndarray)):   
            X=np.array(X)
        
        m=len(X)
        
        if (parameter==0):
            augmented_X=X.reshape(m,32,32)
            return augmented_X
        elif(parameter==1):
            augmented_X=X.reshape(m,4,16,16)
            return augmented_X
       

#PARAMETERS
# number_of_filters = 4
# filter_dim = 3
# input_dim = 32
# padding = 1
# stride = 2

# TESTING
inputstr="1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2"
inputlist=inputstr.split(',')
inputlist=[int(i) for i in inputlist]
inputlist=inputlist*32
inputlist=inputlist+[2] #add label
# inputvect=np.array(inputlist)
# inputdata=inputvect.reshape(32,32)
inputdata=np.array(inputlist)

inputdata2=np.array([inputdata,inputdata])
# print(inputdata2)

cnet=CNN1()
# print(cnet.compute_output(inputdata2))

# a=[[1,0,0],[0,1,0]]
# b=[[0.2,0.01,0.01],[0.01,0.9,0.4]]
# print(cnet.cost(b,a))

cnet.train(inputdata2)