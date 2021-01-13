# Name      :   Pranav Bhandari
# Student ID:   1001551132
# Date      :   09/28/2020

import sys, numpy as np, math

def get_phi(features, degree):
    phi = []
    for row in features:
        phi_values = []
        phi_values.append(1)
        for num in row:
            for i in range(degree):
                phi_values.append(pow(num, i+1))
        phi.append(phi_values)
    return phi

def training(training_file, degree, lambda_value):
    file = open(training_file, "r")
    numbers_array = np.array([[float(x) for x in line.split()] for line in file])
    t = numbers_array[:, -1]
    features = numbers_array[:, :-1]
    num_features = len(features[0])
    M = 1 + num_features*degree
    phi = get_phi(features, degree)
    lambda_I = np.multiply(np.identity(M, dtype = float), lambda_value)
    phi_transpose = np.transpose(phi)
    inverse = np.linalg.pinv(np.add(lambda_I, np.dot(phi_transpose, phi)))

    w = np.dot(np.dot(inverse, phi_transpose), t)
    for i in range(len(w)):
        print("w{}={:.4f}".format(i, w[i]))
    return w

def test(w, degree, test_file):
    file = open(test_file, "r")
    numbers_array = np.array([[float(x) for x in line.split()] for line in file])
    features = numbers_array[:, :-1]
    w_transpose = np.transpose(w)
    phi = get_phi(features, degree)
    for i in range(len(features)):
        output = np.dot(w_transpose, phi[i])
        target_value = numbers_array[i][-1]
        squared_error = pow(target_value-output, 2)
        print("ID={:5d}, output={:14.4f}, target value = {:10.4f}, squared error = {:.4f}".format(i+1, output, target_value, squared_error))

def linear_regression(training_file, test_file, degree, lambda_value):
    w = training(training_file, degree, lambda_value)
    test(w, degree, test_file)

if __name__ == '__main__':
    linear_regression(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))