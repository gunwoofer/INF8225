import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()
X = digits.data
y = digits.target

y_one_hot = np.zeros((y.shape[0],len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]),y] = 1 # one hot target or shape N x K

X_train,X_test,y_train,y_test = train_test_split(X,y_one_hot,test_size = 0.3,random_state = 42)

X_test,X_validation,y_test,y_validation = train_test_split(X_test,y_test,test_size = 0.5,random_state = 42)



def softmax(x):
    e_x = np.exp(x- max(x))
    return e_x / e_x.sum()


def get_accuracy(X,y,W):
    good_pred = 0
    bad_pred = 0
    for digit, correct in zip(X, y):
        y_pred = softmax(W.dot(digit))
        index_max = np.argmax(y_pred)
        if (correct[index_max] == 1):
            good_pred += 1
        else:
            bad_pred += 1
    return good_pred / (good_pred + bad_pred) * 100


def get_grads(y,y_pred,X):
    gradient =  y.reshape(y.shape[0], 1).dot(X.reshape(1, X.shape[0])) \
    - y_pred.reshape(y_pred.shape[0], 1).dot(X.reshape(1, X.shape[0]))
    return gradient
    
# Negative likelihood
def get_loss(y,y_pred):
    return np.log(np.dot(y, y_pred))

def partie2(minibatch_size, lr):
    best_W = None
    best_accuracy = 0
    nb_epochs = 50
    lossesTrain = []
    lossesValid = []
    accuracies = []
    W = np.random.normal(0,0.01,(len(np.unique(y)),X.shape[1])) # weights of shape K x L
    biais = np.zeros((10,1))

    for epoch in range(nb_epochs):
        lossTrain = 0
        lossValid = 0
        accuracy = 0
        gradW = 0
        for i in range(0,X_train.shape[0]):
            x = X_train[i].reshape((64, 1))
            # Compute the prediction
            y_pred = softmax(W.dot(x) + biais).squeeze()
            gradW -= get_grads(y_train[i], y_pred, X_train[i])
            
            # We update our W every minibatch
            if (i != 0 and i % minibatch_size == 0):
                gradW /= minibatch_size
                W -= lr * gradW
                grad = 0

        # loss on train
        for i in range(0, X_train.shape[0]):
            x = X_train[i].reshape((64, 1))
            y_pred = softmax(W.dot(x) + biais)
            lossTrain -= get_loss(y_train[i], y_pred) 
        lossesTrain.append(lossTrain / X_train.shape[0]) 

        # loss on validation
        for i in range(0, X_validation.shape[0]):
            x = X_validation[i].reshape((64, 1))
            y_pred = softmax(W.dot(x) + biais)
            lossValid -= get_loss(y_validation[i], y_pred) 
        lossesValid.append(lossValid / X_validation.shape[0]) 

        # compute the accuracy on the validation set
        accuracy = get_accuracy(X_validation, y_validation, W)
        accuracies.append(accuracy) 
        # select the best parameters based on the validation accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W = W
            

    accuracy_on_unseen_data = get_accuracy(X_test,y_test,best_W)
    print(accuracy_on_unseen_data) 

    
    plt.plot(lossesTrain, label="train")
    plt.plot(lossesValid, label="validation")
    plt.legend()
    plt.show()

    plt.imshow(best_W[4,:].reshape(8,8))
    plt.show()

# Question 2.a
lrs = [0.1, 0.01, 0.001]
tailles_minibatchs = [20, 89, 200, 1000]

for lr in lrs:
  for taille_minibatch in tailles_minibatchs:
    # On ne peut pas compute pour ces valeurs (on obtient un x trop proche de 0 dans log(x))
    if(taille_minibatch == 1 and lr == 0.1):
        pass
    else:
        print("learning rate : ", lr)
        print("taille minibash : ", taille_minibatch)
        partie2(taille_minibatch, lr)

