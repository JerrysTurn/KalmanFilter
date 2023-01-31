import numpy as np
import matplotlib.pyplot as plt

# t range from -4 ~ 4
t = np.arange(-100, 101) / 25

# generate xy measurement noise error
noise_x = np.random.randn(201) / 4
noise_y = np.random.randn(201) / 4

# true trajectory
real_x = t**2
real_y = t**2 - 2*t + 3

# generate xy measurement data 
measure_x = real_x + noise_x
measure_y = real_y + noise_y

# Kalman filter for 2dimensional space
class KalmanFilter():

    # dt : time period of measurement
    # process_noise : acceleration standard deviation , measurement_noise : error standard deviation
    # input value format have to be value or list
    def __init__(self, dt, process_noise, measurement_noise, initial_state_mean, initial_state_covariance):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_mean = np.array(initial_state_mean)
        self.state_covariance = np.array(initial_state_covariance)

        # constant acceleration / transition matrix : F
        self.transition_matrix = np.array([[1, dt, (dt**2)/2, 0,  0,         0],
                                           [0,  1,        dt, 0,  0,         0],
                                           [0,  0,         1, 0,  0,         0],
                                           [0,  0,         0, 1, dt, (dt**2)/2],
                                           [0,  0,         0, 0,  1,        dt],
                                           [0,  0,         0, 0,  0,         1]])

        # process_noise_matrix : Q
        self.process_noise_matrix = np.array([[(dt**4)/4, (dt**3)/2, (dt**2)/2,         0,         0,         0], 
                                              [(dt**3)/2,     dt**2,        dt,         0,         0,         0],
                                              [(dt**2)/2,        dt,         1,         0,         0,         0],
                                              [        0,         0,         0, (dt**4)/4, (dt**3)/2, (dt**2)/2],
                                              [        0,         0,         0, (dt**3)/2,     dt**2,        dt],
                                              [        0,         0,         0, (dt**2)/2,        dt,         1]]) * (process_noise**2)

        # measurement uncertainty by measurement noise : R
        self.measurement_uncertainty = np.array([[measurement_noise**2,                    0],
                                                 [                   0, measurement_noise**2]])

        # measurement matrix : H
        self.measurement_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0]])

        # transition
        print(self.transition_matrix)
        print(self.process_noise_matrix)
        print(self.measurement_uncertainty)

    def predict(self):
        # @ : matrix multiplication
        self.state_mean = self.transition_matrix @ self.state_mean 
        self.state_covariance = self.transition_matrix @ self.state_covariance @ np.transpose(self.transition_matrix) + self.process_noise_matrix

    def update(self, measurement):
        # measurement_prediction : Hx
        measurement_prediction = np.dot(self.measurement_matrix, self.state_mean)
        # measur_predict_diff : z - Hx
        measure_predict_diff = measurement - measurement_prediction
        # temp : HPH' + R
        temp = self.measurement_matrix @ self.state_covariance @ np.transpose(self.measurement_matrix) + self.measurement_uncertainty
        # kalman gain calculation
        kalman_gain = self.state_covariance @ np.transpose(self.measurement_matrix) @ np.linalg.inv(temp)
        # update estimate with measurement
        self.state_mean = self.state_mean + kalman_gain @ measure_predict_diff
        # update covariance matrix
        self.state_covariance = np.dot((np.eye(np.shape(self.state_covariance)[0]) - np.dot(kalman_gain, self.measurement_matrix)), self.state_covariance)

        return self.state_mean[0], self.state_mean[3]


# dt : 0.04, process_noise : 0.3 (standard derivation of process noise matrix)
# measurement_noise = 0.1 

initial_mean = [0, 0, 0, 0, 0, 0]
initial_covariance = np.eye(6) * 500
test = KalmanFilter(0.04, 0.3, 0.1, initial_mean, initial_covariance)

# input : X_center, Y_center (measurement)
x_predict = []
y_predict = []
for i in range(len(t)):
    test.predict()
    temp_x, temp_y = test.update(np.array([measure_x[i], measure_y[i]]))
    x_predict.append(temp_x)
    y_predict.append(temp_y)

# output : real center (prediction)

# plot input value
plt.plot(measure_x, measure_y)
plt.plot(x_predict, y_predict)
plt.grid()
plt.show()