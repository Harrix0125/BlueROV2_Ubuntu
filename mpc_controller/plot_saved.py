import numpy as np
import matplotlib.pyplot as plt

# Load the saved file
data = np.load('BB_home_test.npz')

rov_x = data['real_target_data_x']
rov_y = data['real_target_data_y']


plt.figure(figsize=(10, 8))
plt.plot(rov_x, rov_y, label="BlueBoat Trajectory", color='blue')


plt.legend()
plt.title("BlueBoat vs Target Trajectory")
plt.grid(True)
plt.show()