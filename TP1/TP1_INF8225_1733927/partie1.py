import numpy as np
import matplotlib.pyplot as plt


prob_pluie = np.array([0.8,0.2]).reshape(2,1,1,1)
print("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))

prob_arroseur = np.array([0.9,0.1]).reshape(1,2,1,1)
print("Pr(Arroseur)={}\n".format(np.squeeze(prob_arroseur)))

watson = np.array([[0.8,0.2],[0,1]]).reshape(2,1,2,1)
print("Pr(Watson|Pluie)={}\n".format(np.squeeze(watson)))

holmes = np.array([[1,0], [0.1,0.9], [0,1], [0,1]]).reshape(2,2,1,2)
print("Pr(Holmes|Pluie,arroseur)={}\n".format(np.squeeze(holmes)))


print ("-----------")

print ("a)")
print("Pr(Watson)={}\n".format((watson*prob_pluie).sum(0).squeeze()[1]))

print ("b)")
prob_holmes = ((holmes*prob_pluie*prob_arroseur).sum(0).sum(0).squeeze())
watson_holmes = ((watson*prob_pluie*holmes*prob_arroseur).sum(0).sum(0).squeeze())
print("Pr(Watson|Pluie)={}\n".format((watson_holmes / prob_holmes)[1][1]))

print ("c)")
num = ((prob_pluie*watson*holmes*prob_arroseur).sum(0))
den = ((prob_pluie*holmes*prob_arroseur).sum(0))
print("Pr(Watson|H,A)={}\n".format((num / den)[0][1][1]))

print ("d)")
somme = (watson*prob_pluie*holmes).sum(0).sum(2)
print("Pr(Watson|A)={}\n".format(somme[0][1]))

print ("e)")
print("Pr(Watson|P)={}\n".format(watson.squeeze()[1][1]))

