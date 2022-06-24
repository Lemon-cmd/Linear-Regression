import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt 

from mnist import MNIST
mndata = MNIST('.')

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

to_np_array = lambda x, shape: np.array(x).reshape(shape)

# convert to numpy arrays
x_train = to_np_array(x_train, [-1, 28 * 28])
y_train = to_np_array(y_train, [-1, 1])

x_test = to_np_array(x_test, [-1, 28 * 28])
y_test = to_np_array(y_test, [-1, 1])

""" 
    Convert the Multi-Classification Problem into Binary Classification
    Use -1.0 instead of 0.0 as it allows gradient descent (i.e., no zero-gradients)
"""
y_train = np.where(y_train != 1.0, -1.0, 1.0)
y_test = np.where(y_test != 1.0, -1.0, 1.0)


class LinearReg:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        self.bias = bias 
        
        self.A = npr.uniform(-0.2, 0.2, [out_dim, in_dim])
        
        if bias:
            self.B = np.ones([out_dim, 1], dtype=np.float64)
            
            
    def __call__(self, x, apply_sign=False):
        yh = self.A @ x 
        if self.bias:
            yh += self.B
            
        if apply_sign:
            return np.sign(yh)
        return yh


    def loss_fn(self, yh, y):
        return (yh - y) ** 2.0
    

def train(model: LinearReg, train_data: tuple, epochs: int = 2, verbose: bool = False, train_with_bias: bool = False):
    assert len(train_data) == 2
    
    lrate = 9e-9
    
    if train_with_bias:
        assert model.bias == True, 'The bias parameter is not enabled.'
    
    N = train_data[0].shape[0]
    for epoch in range(epochs):
        loss = 0.0
        dJdA = np.zeros_like(model.A, dtype=np.float64)
        
        for i in range(N):
            # x : d x 1 and y : d-tilda x 1
            x = train_data[0][i][..., None]
            y = train_data[1][i][..., None]
            
            # forward propagation: compute yh
            yh = model(x)
            loss += model.loss_fn(yh, y)
                        
            # compute gradients
            dJdyh = (yh - y)           # d-tilda x 1
            dyhdA = x                  # d x 1
            
            # A is of the shape d-tilda x d
            dJdA += dJdyh @ dyhdA.T
            
            if train_with_bias:
                """ Calculate dJ w.r.t Bias or B, i.e., dJ/dB """
                # tip: djdyh is used here.
                # reminder: the loss function J = (1 / (2 * N)) summation of (yh - y)^2
                
                # remove continue once you are done
                continue

        loss /= (N * 2.0)
        dJdA /= N
        
        # update A 
        model.A -= lrate * dJdA
        
        if train_with_bias:
            """ Update Bias --- declare dJdB somewhere and make sure the shape is correct...."""
            # what should we with dJdB before updating B? 
            # hint: Look at line 90
            # ask yourself, why we need to do that
            
            #model.B -= lrate * dJdB
        
        if verbose: 
            print("Training iteration:", epoch + 1, "Loss:", float(loss))
        if epoch == epochs - 1:
            print("Training iteration:", epoch + 1, "Loss:", float(loss))
    
    return model 

def evaluate(model: LinearReg, test_data: tuple):
    assert len(test_data) == 2
    
    loss, accuracy = 0.0, 0.0
    N = test_data[0].shape[0]
    
    for i in range(N):
        # x : d x 1 and y : d-tilda x 1
        x = test_data[0][i][..., None]
        y = test_data[1][i][..., None]
            
        # forward propagation: compute yh
        yh = model(x)
        loss += model.loss_fn(yh, y)
        
        # calculate accuracy
        yh = np.sign(yh)
        accuracy += np.where(yh == y, 1.0, 0.0).sum() / y.size
        
    loss /= (N * 2.0)
    accuracy /= N
    print("Loss:", float(loss), "Accuracy:", accuracy)

model = LinearReg(in_dim = 784, out_dim = 1, bias = False)

# don't over train it ---- keep no. of epochs minimal
model = train(model, (x_train, y_train), epochs = 15)   

evaluate(model, (x_test, y_test)) 
print("Pred:", model(x_test[0][..., None], apply_sign=True), "True:", y_test[0])
