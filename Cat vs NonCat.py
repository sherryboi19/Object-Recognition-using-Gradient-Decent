
import cv2 as cv
import numpy as np

# Making Dataset X
X=[]
for i in range(209):    
    img=cv.imread("D:\\Images\\"+str(i)+".jpg")
    img=np.array(img)
    img2=img.reshape(1,-1)
    X.append(img2)

# Converting List to np array
X=np.array(X)    
X=X.reshape(209,-1)

# Scalling X
X=X/255

# Different Parameters
m,c = X.shape
iterations  =10000
alpha  =0.001
theta = np.zeros((c+1,1))

# Adding Bised Term
ones = np.ones((m,1))
X = np.hstack((ones,X))

# sigmaoid fucntion
def sigmoid(h):
    g = 1/(1+np.exp(-h))
    return g

# Cost Fucction 
def Get_cost_J(X,Y,Theta,m):
    
    temp1 = np.multiply(Y,np.log(sigmoid(np.dot(X,Theta))))
    temp2 = np.multiply((1-Y),np.log(1-sigmoid(np.dot(X,Theta))))
    
    J  =(-1/m)*np.sum(temp1+temp2)
    return J

# Grdient Decent
def gradient_decent(x,y,theta,alpha,iterations,m):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        z = np.dot(x,theta)
        predictions = sigmoid(z)        
        error = predictions-y
        slope = (1/m)*np.dot(x.T,error)
        theta = theta  - (alpha*slope)
        history[i] = Get_cost_J(x, y, theta, m)   
    return (theta,history)

# Actual Y 
Y = np.array([0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0])
Y=Y[:,np.newaxis]

# Fitting Model
theta,hist = gradient_decent(X, Y, theta, alpha, iterations, m)

# Making New Predictions
def predict(tempx,theta):
    tempx=cv.resize(tempx,(64,64)) # Resizing to 64X64
    tempx=np.array(tempx)
    tempx=tempx.reshape(1,-1)
    tempx=tempx/255
    tempone = np.ones((1,1))
    tempx = np.hstack((tempone,tempx))
    z = np.dot(tempx,theta)
    predictions = sigmoid(z)
    return predictions

# Testing Image
tx=cv.imread("D:\\cat.jpg")
predicted=predict(tx,theta)

# Printing Result
if(predicted>0.5):
    print("Prediction : Its CAT.")
else:
    print("Prediction : Not a CAT.")
