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

W = np.random.normal(0,0.01,(len(np.unique(y)),X.shape[1])) # weights of shape K x L

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 50
minibatch_size = len(y)//20

losses = []
accuracies = []

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Test de la fonction softmax()
# x = np.array([9, 4, 1, 5])
# print(softmax(x))


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
    
# Test de la fonction get_grads()
# X = np.array([1, 2, 3, 4])
# Y = np.array([2, 4])
# Ypred = np.array([1, 3])
# print(get_grads(Y, Ypred, X))

def get_loss(y,y_pred):
    # Negative likelihood
    return -np.log(np.dot(y, y_pred))

# Test de la fonction get_loss()
# actu = np.array([0, 0, 1, 0])
# pred = np.array([0, 0, 0.99, 0])
# print(get_loss(actu, pred))



for epoch in range(nb_epochs):
    loss = 0
    accuracy = 0
    for i in range(0,X_train.shape[0],minibatch_size):
        pass # TODO
    losses.append(loss) # compute the loss on the train set
    accuracy = 0.01 # TODO
    accuracies.append(accuracy) # compute the accuracy on the validation set
    if accuracy > best_accuracy:
        pass # select the best parameters based on the validation accuracy

accuracy_on_unseen_data = get_accuracy(X_test,y_test,best_W)
print(accuracy_on_unseen_data) # 0.897506925208

plt.plot(losses)

#plt.imshow(best_W[4,:].reshape(8,8))


