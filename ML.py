# This program aims to develop a machine learning library by implementing six machine learning algorithms from scratch:
# 1.Linear Regression (and Polynomial Regression)
# 2.Logistic Regression
# 3.K-Nearest Neighbours (KNN)
# 4.K-Means Clustering
# 5.Decision Trees
# 6.N-Layer Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numbers
import pickle

plt.ion() 

"""This function is responsible for computing the mean square error"""
def Mean_square_cf(f, y, w, l=0.001, regularization=True):
    m = np.size(f, axis=0)
    if regularization: # To turn on or off regularization
        c = np.mean((f - y) ** 2)/2  + (l*np.sum(w**2))/(2*m)
    else:
        c = np.mean((f - y) ** 2)/2
    return c


"""This function is responsible for computing the binary cross entropy error"""
def Binary_entropy_costfunction(f, y, w, l=0.001, regularization=True):
    m = np.size(f, axis=0)
    if regularization: # To turn on or off regularization
        c = (-np.dot(y.T,np.log(f+1e-24))-np.dot((1-y).T, np.log(1-f+1e-24)))/m + (l*np.sum(w**2))/(2*m)
    else:
        c = (-np.dot(y.T,np.log(f+1e-24))-np.dot((1-y).T, np.log(1-f+1e-24)))/m
    return c[0][0]

"""This function is responsible for computing the cross entropy error"""
def Cross_entropy_costfunction(f, y, w, l=0.001, regularization=True):
    m = np.size(y, axis=0)
    ones = (y == 1)
    loss = np.zeros(y.shape)
    loss[ones] = -np.log(f[ones]+1e-18)
    loss = np.sum(loss) / m 
    if regularization: 
        loss += (l * np.sum(w**2)) / (2 * m)
    return loss

def polynomial_regression(x, degree):
    features_count = np.size(x, axis=1)
    new_features = np.zeros((np.size(x,axis=0), np.size(x, axis=1)*degree))
    for i in range(degree):
        for j in range(features_count):
            new_features[:,features_count*i+j] = x[:,j]**(i+1)
    return new_features
            
"""This function implements f = wx + b"""
def Linear(w, x, b):
    f = np.matmul(x,np.reshape(w,(np.size(w, axis=1),1)))+b
    if np.size(f) == 1:
        return f[0]
    else:
        return f

"""This functions computes the sigmoid activation of the given weights and bias"""
def sigmoid_activation(w,x, b):
    f = 1/(1+np.exp(-Linear(w,x,b)))
    return f

"""This functions computes the softmax activation of the given weights and bias"""
def softmax_activation(activations):
    softmax_activations = np.exp(activations)
    softmax_activations = softmax_activations/np.sum(softmax_activations)
    return softmax_activations

"""This functions computes the ReLU activation of the given weights and bias"""
def ReLU_activation(w,x, b):
    f = Linear(w,x,b)
    _sign =  f < 0
    f[_sign] = 0
    return f


"""This function is responsible for handling the gradient descent algorithm"""
def GD(Activation_function, Cost_function, x, xtest, xcv, y, ytest, ycv, W_in, B_in, cutoff_cost, iterations, l_regularization, learning_rate, b_learning_rate, weights_file, bias_file, print_interval, optimization="None"):
    w = W_in
    b = B_in
    m = np.size(y)
    cost = Cost_function(Activation_function(w, x, b), y, w, l_regularization)
    i = 0
    k = 0.01 #Adaptive factor
    beta1 = 0.9
    beta2 = 0.99
    cost_history = []
    test_history = []
    cv_history = []
    grad_history = []
    b_grad_history = []
    print(i,":",cost)
    cost_history.append(cost)
    grad_history.append(np.matmul(np.reshape((Activation_function(w, x, b)-y), (1,m)),x))
    b_grad_history.append(np.sum(Activation_function(w, x, b)-y))
    Vdw = np.zeros(w.shape)
    Vdb = 0
    Sdw = np.zeros(w.shape)
    Sdb = 0
    figure, axis = plt.subplots(1, 3)
    while cost>cutoff_cost and iterations>i:
        grad_current = np.matmul(np.reshape((Activation_function(w, x, b)-y), (1,m)),x)/m #Calculation of the gradient at the point
        b_grad_current = np.sum(Activation_function(w, x, b)-y)
        
        if optimization == "Adam":
            # Rms prop
            Sdw = beta2*Sdw + (1-beta2) * grad_current ** 2
            Sdb = beta2*Sdb + (1-beta2) * b_grad_current ** 2

            #Calculating the momentum
            Vdw = beta1*Vdw + (1-beta1) * grad_current 
            Vdb = beta1*Vdb + (1-beta1) * b_grad_current
            
            Sdw_corrected = Sdw/(1-beta2**(i+1))
            Sdb_corrected = Sdb/(1-beta2**(i+1))
            Vdw_corrected = Vdw/(1-beta1**(i+1))
            Vdb_corrected = Vdb/(1-beta1**(i+1))


            w,b = w - learning_rate*(Vdw_corrected/(np.sqrt(Sdw_corrected)+1e-24)), b - b_learning_rate*(Vdb_corrected/(np.sqrt(Sdb_corrected) + 1e-24))

            cost = Cost_function(Activation_function(w, x, b), y, w, l_regularization) # Cost calculation

            i+=1
            if i%print_interval == 0:
                    cost_history.append(cost)
                    test_history.append(Cost_function(Activation_function(w, xtest, b), ytest, w, 0))
                    cv_history.append(Cost_function(Activation_function(w, xcv, b), ycv, w, 0))
                    print(i,":",cost, "w Grad_current:",np.abs(grad_current)<np.abs(grad_history[-1]), "B Grad_current:", b_grad_current<abs(b_grad_history[-1]))
                    w_df = pd.DataFrame(w)
                    b_df = pd.DataFrame([b])
                    w_df.to_csv(weights_file, index=False)
                    b_df.to_csv(bias_file, index=False)
                    plot_data([cost_history, test_history, cv_history], [f"{print_interval} Iterations", f"{print_interval} Iterations", f"{print_interval} Iterations"], ["Cost", "Cost", "Cost"], ["Training Data", "Test Data", "Cross Validation Data"], axis)

            grad_history.append(grad_current)
            b_grad_history.append(b_grad_current)
        
        elif optimization == "GD_momentum":
            #Calculating the momentum
            Vdw = beta1*Vdw + (1-beta1) * grad_current 
            Vdb = beta1*Vdb + (1-beta1) * b_grad_current

            w,b = w - learning_rate*Vdw, b - b_learning_rate*Vdb #Gradient descend with momentum is applied to the weights and biases

            cost = Cost_function(Activation_function(w, x, b), y, w, l_regularization) # Cost calculation

            i+=1
            if i%print_interval == 0:
                    cost_history.append(cost)
                    test_history.append(Cost_function(Activation_function(w, xtest, b), ytest, w, 0))
                    cv_history.append(Cost_function(Activation_function(w, xcv, b), ycv, w, 0))
                    print(i,":",cost, "w Grad_current:",np.abs(grad_current)<np.abs(grad_history[-1]), "B Grad_current:", b_grad_current<abs(b_grad_history[-1]))
                    w_df = pd.DataFrame(w)
                    b_df = pd.DataFrame([b])
                    w_df.to_csv(weights_file, index=False)
                    b_df.to_csv(bias_file, index=False)
                    plot_data([cost_history, test_history, cv_history], [f"{print_interval} Iterations", f"{print_interval} Iterations", f"{print_interval} Iterations"], ["Cost", "Cost", "Cost"], ["Training Data", "Test Data", "Cross Validation Data"], axis)

            grad_history.append(grad_current)
            b_grad_history.append(b_grad_current)
        
        elif optimization == "adaptive_learning":
            w,b = w - learning_rate*grad_current, b - b_learning_rate*b_grad_current #Gradient descend is applied to the weights and biases

            cost = Cost_function(Activation_function(w, x, b), y, w, l_regularization) # Cost calculation

            # This part of the coding is responsible for logging the current progress of the code and saving the weights and biases
            # so that progress is not lost
            i+=1
            if i%print_interval == 0:
                cost_history.append(cost)
                test_history.append(Cost_function(Activation_function(w, xtest, b), ytest, w, 0))
                cv_history.append(Cost_function(Activation_function(w, xcv, b), ycv, w, 0))
                print(i,":",cost, "w Grad_current:",np.abs(grad_current)<np.abs(grad_history[-1]), "B Grad_current:", b_grad_current<abs(b_grad_history[-1]), "aw:", learning_rate, "ba:", b_learning_rate)
                w_df = pd.DataFrame(w)
                b_df = pd.DataFrame([b])
                w_df.to_csv(weights_file, index=False)
                b_df.to_csv(bias_file, index=False)
                plot_data([cost_history, test_history, cv_history], [f"{print_interval} Iterations", f"{print_interval} Iterations", f"{print_interval} Iterations"], ["Cost", "Cost", "Cost"], ["Training Data", "Test Data", "Cross Validation Data"], axis)

            grad_history.append(grad_current)
            b_grad_history.append(b_grad_current)

            #This try-except block is just present so that an error does not occur when the number of iterations is not greater than 10000
            try:

                #This part implements adaptive learning rate which ensures a faster training speed of the model
                sign_change = np.sign(grad_history[-2]) != np.sign(grad_history[-1]) # This line checks if the gradient is oscillating 
                learning_rate[sign_change] *= 0.0001                                 # If it is oscillating, it means the learning rate is too large and is decreased
                learning_rate[~sign_change] *= 1+k                                   # If the sign does not change, it means the learning rate can be increased for faster learning
                learning_rate = np.clip(learning_rate, 1e-24, 1)

                if np.sign(b_grad_history[-2]) != np.sign(b_grad_history[-1]):
                    b_learning_rate *= 0.0001
                else:
                    b_learning_rate *= 1+k
                b_learning_rate = np.clip(b_learning_rate, 1e-24, 1)

                #These conditions are present to ensure that the model does not diverge
                #and that the learning rate is sufficiently managed
                if cost_history[-5] <= cost_history[-1]:
                    break

            except:
                pass

        elif optimization == "None":
            w,b = w - learning_rate*grad_current, b - b_learning_rate*b_grad_current #Gradient descend is applied to the weights and biases

            cost = Cost_function(Activation_function(w, x, b), y, w, l_regularization) # Cost calculation

            # This part of the coding is responsible for logging the current progress of the code and saving the weights and biases
            # so that progress is not lost
            i+=1
            if i%print_interval == 0:
                cost_history.append(cost)
                test_history.append(Cost_function(Activation_function(w, xtest, b), ytest, w, 0))
                cv_history.append(Cost_function(Activation_function(w, xcv, b), ycv, w, 0))
                print(i,":",cost)
                w_df = pd.DataFrame(w)
                b_df = pd.DataFrame([b])
                w_df.to_csv(weights_file, index=False)
                b_df.to_csv(bias_file, index=False)
                plot_data([cost_history, test_history, cv_history], [f"{print_interval} Iterations", f"{print_interval} Iterations", f"{print_interval} Iterations"], ["Cost", "Cost", "Cost"], ["Training Data", "Test Data", "Cross Validation Data"], axis)

        else:
            raise Exception("Invalid entry for optimization") 

    return w,b, cost_history


"""This function is solely responsible for loading the data as a pandas data frame from a given file"""
def load_data(f, rows, sample_rows):
    if rows == "all":
        data_sample = pd.read_csv(f)
    else:
        data = pd.read_csv(f, nrows = rows)
        data_sample = data.sample(sample_rows).to_numpy()
    return data_sample


"""This function is solely responsible for ploting the data and for visualization"""
def plot_data(data, x_axis_label, y_axis_label, title, axis):
    for i in range(len(title)):
        axis[i].clear() 
        axis[i].plot(data[i], color='g')
        axis[i].set(xlabel=x_axis_label[i], ylabel=y_axis_label[i])
        axis[i].set_title(title[i])
    plt.draw()
    plt.pause(0.25)  

def Linear_regression_main(data_tr, data_test, data_cv, w, b, degree):
    xcv = polynomial_regression(data_cv[:,:-1],degree)
    ycv = np.reshape(np.array([data_cv[:,-1]]),(np.size(xcv, axis=0),1))
    xt = polynomial_regression(data_test[:,:-1],degree)
    yt = np.reshape(np.array([data_test[:,-1]]),(np.size(xt, axis=0),1))
    x = polynomial_regression(data_tr[:,:-1],degree)
    y = np.reshape(np.array([data_tr[:,-1]]),(np.size(x, axis=0),1))

    w, b ,cj= GD(Linear, Mean_square_cf, x,xt, xcv, y, yt, ycv, w, b, 0, 6000,0.01,np.full((1,np.size(x, axis=1)),1e-13),1e-13, r"Linear_regression\weights.csv", r"Linear_regression\bias.csv", 100, optimization="adaptive_learning")

    w_df = pd.DataFrame(w)
    b_df = pd.DataFrame([b])
    w_df.to_csv(r'Linear_regression\weights.csv', index=False)
    b_df.to_csv(r'Linear_regression\bias.csv', index=False)

    
def Logistic_regression_main(data, w, b):    
    x = data[:,:-1]
    y = np.reshape(np.array([data[:,-1]]),(np.size(x, axis=0),1))

    print(x, y, w, b)
    #w, b ,cj= GD(sigmoid_activation, Binary_entropy_costfunction, x,y, w, b, 0.1, 2500,0.001,np.full((1,np.size(x, axis=1)),1e-24),1e-24, r"Logistic_regression\weights.csv", r"Logistic_regression\bias.csv", 100, optimization="adaptive_learning")

    w_df = pd.DataFrame(w)
    b_df = pd.DataFrame([b])
    w_df.to_csv(r'Logistic_regression\weights.csv', index=False)
    b_df.to_csv(r'Logistic_regression\bias.csv', index=False)

def activation_function_derivative(Activation_func, J):
        if Activation_func == sigmoid_activation:
            f = 1/(1+np.exp(-J)) + 1e-24
            return f*(1-f)
        elif Activation_func == ReLU_activation:
            _sign = J > 0
            f = np.zeros(J.shape)
            f[_sign] = 1
            return f
        elif Activation_func ==  Linear:
            return np.full(J.shape, 1)
        elif Activation_func == softmax_activation:
            raise Exception("Softmax derivative is already included")
        else:
            raise Exception("Invalid Activation function encountered !")

"""This class defines a single neural layer present in the neural network"""
class Neural_layer:
    def __init__(self, weights_layer, bias, Activation_func, learning_rate = 1e-4, optimization="None"):
        self.weights = weights_layer
        self.bias = bias
        self.activation_func = Activation_func
        self.node_number = self.weights.shape[0]
        self.activation_current = np.zeros((self.node_number, 1))
        self.optimization = optimization


        self.w_learning_rate = np.full(self.weights.shape, learning_rate)
        self.b_learning_rate = np.full(self.bias.shape, learning_rate)
        self.dCo_wl = np.zeros(self.weights.shape)
        self.dCo_bl = np.zeros(self.bias.shape)
        self.Vdw = np.zeros(self.weights.shape)
        self.Vdb = np.zeros(self.bias.shape)
        self.Sdw = np.zeros(self.weights.shape)
        self.Sdb = np.zeros(self.bias.shape)


        self.grad_history = []
        self.b_grad_history = []


    def GD(self, current_iteration):
        i = current_iteration
        k = 0.01 #Adaptive factor
        beta1 = 0.9
        beta2 = 0.99
        
        if self.optimization == "Adam":
            # Rms prop
            self.Sdw = beta2*self.Sdw + (1-beta2) * self.dCo_wl ** 2
            self.Sdb = beta2*self.Sdb + (1-beta2) * self.dCo_bl ** 2

            #Calculating the momentum
            self.Vdw = beta1*self.Vdw + (1-beta1) * self.dCo_wl 
            self.Vdb = beta1*self.Vdb + (1-beta1) * self.dCo_bl
            Sdw_corrected = self.Sdw/(1-beta2**(i+1))
            Sdb_corrected = self.Sdb/(1-beta2**(i+1))
            Vdw_corrected = self.Vdw/(1-beta1**(i+1))
            Vdb_corrected = self.Vdb/(1-beta1**(i+1))

            self.weights = self.weights - self.w_learning_rate*(Vdw_corrected/(np.sqrt(Sdw_corrected)+1e-24))
            self.bias = self.bias - self.b_learning_rate*(Vdb_corrected/(np.sqrt(Sdb_corrected) + 1e-24))
            
            self.grad_history.append(self.dCo_wl)
            self.b_grad_history.append(self.dCo_bl)

        elif self.optimization == "Adam_with_adaptive":
            # Rms prop
            self.Sdw = beta2*self.Sdw + (1-beta2) * self.dCo_wl ** 2
            self.Sdb = beta2*self.Sdb + (1-beta2) * self.dCo_bl ** 2

            #Calculating the momentum
            self.Vdw = beta1*self.Vdw + (1-beta1) * self.dCo_wl 
            self.Vdb = beta1*self.Vdb + (1-beta1) * self.dCo_bl
            Sdw_corrected = self.Sdw/(1-beta2**(i+1))
            Sdb_corrected = self.Sdb/(1-beta2**(i+1))
            Vdw_corrected = self.Vdw/(1-beta1**(i+1))
            Vdb_corrected = self.Vdb/(1-beta1**(i+1))

            self.weights = self.weights - self.w_learning_rate*(Vdw_corrected/(np.sqrt(Sdw_corrected)+1e-24))
            self.bias = self.bias - self.b_learning_rate*(Vdb_corrected/(np.sqrt(Sdb_corrected) + 1e-24))
            
            self.grad_history.append(self.dCo_wl)
            self.b_grad_history.append(self.dCo_bl)
            
            #This try-except block is just present so that an error does not occur when the number of iterations is not greater than 10000
            try:
                #This part implements adaptive learning rate which ensures a faster training speed of the model
                sign_change = np.sign(self.grad_history[-2]) != np.sign(self.grad_history[-1]) # This line checks if the gradient is oscillating 
                self.w_learning_rate[sign_change] *= 0.0001                                 # If it is oscillating, it means the learning rate is too large and is decreased
                self.w_learning_rate[~sign_change] *= 1+k                                   # If the sign does not change, it means the learning rate can be increased for faster learning
                self.w_learning_rate = np.clip(self.w_learning_rate, 1e-24, 1)
                sign_change = np.sign(self.b_grad_history[-2]) != np.sign(self.b_grad_history[-1]) # This line checks if the gradient is oscillating 
                self.b_learning_rate[sign_change] *= 0.0001                                 # If it is oscillating, it means the learning rate is too large and is decreased
                self.b_learning_rate[~sign_change] *= 1+k                                   # If the sign does not change, it means the learning rate can be increased for faster learning
                self.b_learning_rate = np.clip(self.b_learning_rate, 1e-24, 1)

            except:
                pass

        elif self.optimization == "GD_momentum":
            #Calculating the momentum
            self.Vdw = beta1*self.Vdw + (1-beta1) * self.dCo_wl 
            self.Vdb = beta1*self.Vdb + (1-beta1) * self.dCo_bl

            #Gradient descend with momentum is applied to the weights and biases
            self.weights = self.weights - self.w_learning_rate*self.Vdw
            self.bias = self.bias - self.b_learning_rate*self.Vdb 

            self.grad_history.append(self.dCo_wl)
            self.b_grad_history.append(self.dCo_bl)
        elif self.optimization == "adaptive_learning":
            #Gradient descend is applied to the weights and biases
            self.weights = self.weights - self.w_learning_rate*self.dCo_wl
            self.bias = self.bias -self.b_learning_rate*self.dCo_bl
            # This part of the coding is responsible for logging the current progress of the code and saving the weights and biases
            # so that progress is not lost
            self.grad_history.append(self.dCo_wl)
            self.b_grad_history.append(self.dCo_bl)

            #This try-except block is just present so that an error does not occur when the number of iterations is not greater than 10000
            try:
                #This part implements adaptive learning rate which ensures a faster training speed of the model
                sign_change = np.sign(self.grad_history[-2]) != np.sign(self.grad_history[-1]) # This line checks if the gradient is oscillating 
                self.w_learning_rate[sign_change] *= 0.0001                                 # If it is oscillating, it means the learning rate is too large and is decreased
                self.w_learning_rate[~sign_change] *= 1+k                                   # If the sign does not change, it means the learning rate can be increased for faster learning
                self.w_learning_rate = np.clip(self.w_learning_rate, 1e-24, 1)
                sign_change = np.sign(self.b_grad_history[-2]) != np.sign(self.b_grad_history[-1]) # This line checks if the gradient is oscillating 
                self.b_learning_rate[sign_change] *= 0.0001                                 # If it is oscillating, it means the learning rate is too large and is decreased
                self.b_learning_rate[~sign_change] *= 1+k                                   # If the sign does not change, it means the learning rate can be increased for faster learning
                self.b_learning_rate = np.clip(self.b_learning_rate, 1e-24, 1)

            except:
                pass
        elif self.optimization == "None":
            #Gradient descend is applied to the weights and biases
            self.weights = self.weights - self.w_learning_rate*self.dCo_wl
            self.bias = self.bias - self.b_learning_rate*self.dCo_bl
            self.grad_history.append(self.dCo_wl)
            self.b_grad_history.append(self.dCo_bl)
        else:
            raise Exception("Invalid entry for optimization") 


    """This function defines the activation of the neural layer"""
    def activation(self, x):
        if self.activation_func != softmax_activation:
            activation = np.zeros((self.node_number,1))
            x =x.T
            for node in range(self.node_number):
                activation[node, 0] = self.activation_func(np.array([self.weights[node]]), x, self.bias[node][0])
            self.activation_current = activation
        else:
            activation = self.activation_func(np.matmul(self.weights,x)+self.bias)
            self.activation_current = activation
        return activation

"""This class strings together the various neural layers to create the entire neural network"""
class Neural_network:
    def __init__(self, neural_layers, file):
        self.neural_layers = neural_layers
        self.layer_number = len(self.neural_layers)
        self.cost_history = []
        self.test_history = []
        self.accuracy_cv = []
        self.accuracy_train = []
        self.file = file
        if self.neural_layers[-1].activation_func in [Linear,ReLU_activation]:
            self.cost_function = Mean_square_cf
        elif self.neural_layers[-1].activation_func == sigmoid_activation:
            self.cost_function = Binary_entropy_costfunction
        elif self.neural_layers[-1].activation_func == softmax_activation:
            self.cost_function = Cross_entropy_costfunction

    """Gives a summary of the model"""
    def summary(self, verbose=False):
        params = 0
        print("\n---------- Neural Network Summary ----------\n")
        for layer_number in range(len(self.neural_layers)):
            layer = self.neural_layers[layer_number]
            if verbose:
                params += np.size(layer.weights)+np.size(layer.bias)
                print(f"Layer {layer_number} -  Nodes: {layer.node_number} Activation Function: {layer.activation_func} Optimization: {layer.optimization}")
                print(f"Parameters -  Weights: {np.size(layer.weights)} Bias: {np.size(layer.bias)} Total = {np.size(layer.weights)+np.size(layer.bias)}")
                print(f"Weights: {layer.weights} \nbias: {layer.bias}")
                print("")
            else:
                params += np.size(layer.weights)+np.size(layer.bias)
                print(f"Layer {layer_number} -  Nodes: {layer.node_number} Activation Function: {layer.activation_func} Optimization: {layer.optimization}")
                print(f"Parameters -  Weights: {np.size(layer.weights)} Bias: {np.size(layer.bias)} Total = {np.size(layer.weights)+np.size(layer.bias)}")
        print("Total training parameters :", params)


    def save(self):
        print("....Saving neural_network...")
        with open(self.file, "wb") as f:
            pickle.dump(self, f)
        print("....Saved neural_network...")

    """This function is responsible for forward propagation of the network"""
    def predict(self, x):
        prev_layer_activation = x
        for neural_layer in self.neural_layers:
            prev_layer_activation = neural_layer.activation(prev_layer_activation) # This links the activation of one layer to the next
        return prev_layer_activation

    """Computes the derivatives required for back propagation"""
    def Back_prop_derivatives(self, dCo_a_l, Activation_function, weights_layer, biases_layer, activation_prevL):
        if Activation_function == softmax_activation:
            #Since we use softmax with cross_entropy loss
            dCo_wl = np.matmul(dCo_a_l, activation_prevL.T)
            dCo_bl = dCo_a_l
            dCo_a_prevL = np.matmul(weights_layer.T, dCo_a_l)
        else:
            dactive_func = activation_function_derivative(Activation_function, np.matmul(weights_layer, activation_prevL) + biases_layer)
            dCo_wl = np.matmul((dCo_a_l * dactive_func), activation_prevL.T)
            dCo_bl = dCo_a_l * dactive_func
            dCo_a_prevL = np.matmul((dCo_a_l * dactive_func).T, weights_layer).T
        return dCo_wl, dCo_bl, dCo_a_prevL

    """This function is responisble for the back propagation of the network"""
    def Back_prop(self, x, y):
        # The derivative of the cost function is computed depending upon the cost function
        if self.cost_function == Mean_square_cf:
            dCo_a_l = (self.neural_layers[-1].activation_current - y)
        elif self.cost_function == Binary_entropy_costfunction:
            pred = self.neural_layers[-1].activation_current
            dCo_a_l = (pred - y+1e-24)/(pred*(1-pred+1e-24))
            dCo_a_l = np.nan_to_num(dCo_a_l, nan=0, posinf=1e24, neginf=-1e24)
        elif self.cost_function == Cross_entropy_costfunction:
            pred = self.neural_layers[-1].activation_current
            dCo_a_l = pred - y

        # Computing the various derivatives
        for layer_number in range(len(self.neural_layers)-1, -1, -1):
            if layer_number != 0:
                dCo_wl, dCo_bl, dCo_a_l = self.Back_prop_derivatives(dCo_a_l, self.neural_layers[layer_number].activation_func, self.neural_layers[layer_number].weights, self.neural_layers[layer_number].bias , self.neural_layers[layer_number-1].activation_current)
                self.neural_layers[layer_number].dCo_wl = self.neural_layers[layer_number].dCo_wl + dCo_wl
                self.neural_layers[layer_number].dCo_bl = self.neural_layers[layer_number].dCo_bl + dCo_bl
            else:
                dCo_wl, dCo_bl, dCo_a_l = self.Back_prop_derivatives(dCo_a_l, self.neural_layers[layer_number].activation_func, self.neural_layers[layer_number].weights, self.neural_layers[layer_number].bias , x)
                self.neural_layers[layer_number].dCo_wl = self.neural_layers[layer_number].dCo_wl + dCo_wl
                self.neural_layers[layer_number].dCo_bl = self.neural_layers[layer_number].dCo_bl + dCo_bl
                
    """This function is responsible for training the model with the given outputs"""
    def fit(self, mini_batch, x_train, x_test, y_train, y_test, epochs, print_interval, plotting_interval =100, cost_divergence_check=True, verbose = False):
        if verbose: 
            figure, axis = plt.subplots(1, 3)
        m = np.size(x_train, axis = 0)
        for epoch in range(epochs):
            sample = np.random.choice(m, mini_batch, replace=False)
            x  = x_train[sample, :]
            y = y_train[sample, :]
            cost = 0
            for i in range(np.size(x, axis = 0)):
                prediction = self.predict(np.array([x[i]]).T)
                self.Back_prop(np.array([x[i]]).T, np.array([y[i]]).T)
                cost += self.cost_function(prediction, np.array([y[i]]).T, 0, regularization=False)
            cost /= m
            self.cost_history.append(cost)

            if verbose:
                if epoch%plotting_interval == 0:
                    plotting_data = [self.cost_history, self.accuracy_train, self.test_history, self.accuracy_cv]
                    title_data = ["Convergence of Cost function", "Training Accuracy","Test Accuracy"]
                    x_axis_data = ["Iterations", f"{print_interval} Iterations", f"{print_interval} Iterations"]
                    y_axis_data = ["Cost", "Accuracy", "Accuracy"]
                    plot_data(plotting_data, x_axis_data, y_axis_data, title_data, axis)
 
                if epoch%print_interval == 0:
                    print("Current epoch:", epoch)
                    accuracy_test = NN_accuracy_measure(x_test, y_test, self)
                    self.test_history.append(accuracy_test)
                    print("Accuracy test: ", accuracy_test, "%")
                    accuracy_train = NN_accuracy_measure(x, y, self)
                    self.accuracy_train.append(accuracy_train)
                    print("Accuracy Train: ", accuracy_train, "%\n")


            for layer in self.neural_layers:
                layer.dCo_wl /= m
                layer.dCo_bl /= m
                layer.GD(epoch)
                layer.dCo_wl = np.zeros(layer.weights.shape)
                layer.dCo_bl = np.zeros(layer.bias.shape)

            if cost_divergence_check:
                try:
                    if cost > self.cost_history[-100]:
                        print("Cost is diverging.. Try again with different hyperparameters..")
                        break
                    elif cost == self.cost_history[-100]:
                        print("Cost function has hit a minimum... Try again with different model..")
                        break
                except:
                    pass


"""Data cleanup, one hot_encoding and  normalization"""
def  cleanup_data(file, data_columns, rows, sample_rows, one_hot_encoding_column, normalization_factor):
    data = load_data(file, rows, sample_rows)
    data_np = np.array(data)
    m = np.size(data_np[:,0])
    ref = []
    data_cleaned = np.full((m,1), 0.0000,dtype=np.float64)
    means = []
    sD = []
    for i in range(np.size(data_np, axis=1)):
        classes = []
        if i < data_columns-1 or i == one_hot_encoding_column:
            if not isinstance(data_np[0,i], numbers.Number) or i == one_hot_encoding_column:
                for j in range(m):
                    if data_np[j, i] not in classes:
                        classes.append(data_np[j, i])
                one_hot = np.zeros((m, len(classes))) #one_hot encoding
                for category_number in range(len(classes)):
                    _ = (data_np[:,i] == classes[category_number])
                    one_hot[_,category_number] = 1
                data_cleaned = np.append(data_cleaned, one_hot, axis=1)
                means.append("One Hot")
                sD.append("One Hot")
            else:
                rmean = np.mean(data_np[:,i])
                rSD = np.std(data_np[:,i])*normalization_factor
                means.append(rmean)
                sD.append(rSD)
                data_cr = (np.array([data_np[:,i].astype(np.float64)]) - rmean)/rSD #normalization
                data_cleaned = np.append(data_cleaned, data_cr.T, axis=1)
            ref.append(classes)
        else:
            data_cleaned = np.append(data_cleaned, np.array([data_np[:,i].astype(np.float64)]).T, axis=1)
    data_cleaned = data_cleaned[:, 1:]
    return data_cleaned, ref, means, sD

def normalize_data(file, means, sD, normalization_factor):
    data_file = load_data(file, "all", 0)
    data = np.array(data_file)
    m = np.size(data[:,0])
    data_cleaned = np.full((m,1), 0.0000,dtype=np.float64)
    for i in range(np.size(data, axis=1)):
        if means[i] == "One Hot":
            pass
        else:
            data_cr = (np.array([data[:,i].astype(np.float64)]) - means[i])/(sD[i]*normalization_factor) #normalization
            data_cleaned = np.append(data_cleaned, data_cr.T, axis=1)
    data_cleaned = data_cleaned[:, 1:]
    return data_cleaned

def accuracy_measure(x, y,w, b, sample_size, threshold):
    ys = []
    preds = []
    m = np.size(x, axis=0)
    for i in range(sample_size):
        k = random.randint(0,m)
        xt = x[k, :]
        pred = sigmoid_activation(w, xt, b)
        if pred > threshold:
            ys.append(y[k])
            preds.append(pred)
            print("y: ", y[k], "predicted : 1")
        else:
            ys.append(y[k])
            preds.append(pred)
            print("y: ", y[k], "predicted : 0")
    
    correct = 0
    for i in range(len(preds)):
        if int(preds[i]) == int(ys[i]):
            correct += 1

    print("Accuracy :", correct/len(preds)*100, "%")

def NN_accuracy_measure(xt, yt, NN, softmax = True):
    pred = np.zeros(yt.shape)
    for record_number in range(pred.shape[0]):
        prediction = NN.predict(np.array([xt[record_number,:]]).T).T
        if softmax:
            max_index = np.argmax(prediction)
            prediction_s = np.zeros(prediction.shape)
            prediction_s[0, max_index] = 1
            prediction = prediction_s
        pred[record_number] = prediction
    if not softmax:
        ones = pred > 0.5
        pred[ones] = 1
        pred[~ones] = 0
    correct = np.all((pred == yt), axis=1)
    return (np.sum(correct)/np.size(correct, axis=0))*100

"""This function assigns each sample to the closest cluster"""
def find_closest_K(x, K_centroids):
    distance = np.zeros((x.shape[0], K_centroids.shape[0]))
    for k in range(K_centroids.shape[0]):
        distance[:, k] = np.sum((x - K_centroids[k,:])**2, axis=1) #Computes the distance of the sample to each of the cluster centroids
    return np.argmin(distance, axis = 1) # Returns the index of the cluster containing the least distance

"""This function is responsible for computing the new centroids"""
def K_centroid_computing(x, idx, K):
    k_centroids = np.zeros((K, x.shape[1]))
    for k in range(K):
        k_group = (idx == k) #Gets the samples that are assigned to the cluster

        if k_group.sum() == 0:
            k_centroids[k, :] = x[np.random.choice(x.shape[0])] # Computes the new centroid by taking the mean of the samples
        else:
            k_centroids[k, :] = np.mean(x[k_group, :], axis=0) # If there are no samples assigned to the cluster, the centroid is randomly assigned another position

    return k_centroids

"""This function computes the distortion of the given k centroids"""
def compute_distortion(x, idx, k_centroids):
    K = k_centroids.shape[0]
    distortions = np.zeros((x.shape[0], 1))
    for k in range(K):
        k_group = (idx == k)
        distortions[k_group, :] = np.array([np.sum((x[k_group, :] - k_centroids[k,:])**2, axis=1)]).T
    return np.sum(distortions)

"""This is the main K-means clustering algorithm"""
def K_means_clustering(K, x, kk, axis, units, plot = False):
    centroids = []
    colors = ["r", "b", "y", "purple"]
    distortion = np.zeros((1, units))
    for unit in range(units):
        print(f"Unit: {unit} Working ...")
        random_rows = np.random.choice(x.shape[0], K, replace=False)
        k_centroids = x[random_rows]

        idx = find_closest_K(x, k_centroids)
        dist_prev = compute_distortion(x, idx, k_centroids)
        dist_current = 0
        while abs(dist_current - dist_prev) >0: #Checks if convergence has been achieved
            dist_prev = compute_distortion(x, idx, k_centroids)
            idx = find_closest_K(x, k_centroids)
            k_centroids = K_centroid_computing(x, idx, K)
            dist_current = compute_distortion(x, idx, k_centroids)
        

        idx = find_closest_K(x, k_centroids)
        distortion[0,unit] = dist_current
        centroids.append(k_centroids)

        if plot:
            for i in range(x.shape[1]):
                axis[i//kk,i-(i//kk)*kk].clear()
                for j in range(K):
                    axis[i//kk,i-(i//kk)*kk].hist(x[idx==j, i], alpha = 0.5, bins = 100, color = colors[j], label = f"Cluster {j}")
                    axis[i//kk,i-(i//kk)*kk].axvline(k_centroids[j,i], color=colors[j], linestyle='-')
                axis[i//kk,i-(i//kk)*kk].set_title(f"Final Feature {i}")
            plt.pause(0.1)

    centroids_best = centroids[np.argmin(distortion)]
    idx = find_closest_K(x, centroids_best)
    for i in range(x.shape[1]):
        axis[i//kk,i-(i//kk)*kk].clear()
        for j in range(K):
            axis[i//kk,i-(i//kk)*kk].hist(x[idx==j, i], alpha = 0.5, bins = 100, color = colors[j], label = f"Cluster {j}")
            axis[i//kk,i-(i//kk)*kk].axvline(centroids_best[j,i], color=colors[j], linestyle='-')
        axis[i//kk,i-(i//kk)*kk].set_title(f"Final Best Feature {i}")
    plt.pause(0.1)
    return centroids_best, idx
