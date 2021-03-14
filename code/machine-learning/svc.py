#!/usr/bin/env python3

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

x_entry = np.array([[0, 0], [1, 1]])
y_label = np.array([0, 1])

print(x_entry, y_label)

classifier = svm.SVC()
classifier.fit(x_entry, y_label)

# After being fitted, the model can then be used to predict new values:
classifier.predict([[2., 2.]])
# -> array([1])

# get support vectors
support_vectors = classifier.support_vectors_

# get indices of support vectors
indices = classifier.support_

# get number of support vectors for each class
support_vectors_qt = classifier.n_support_

plt.scatter(x_entry[:, 0], x_entry[:, 1], c=y_label, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
x_axis = np.linspace(xlim[0], xlim[1], 30)
y_axis = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(y_axis, x_axis)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')


plt.show()
