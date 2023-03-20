import numpy as np
import matplotlib.pyplot as plt
import csv


headers = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]
data = np.genfromtxt('kc_house_data.csv', delimiter=',', skip_header=1)

    # Extraer la columna sqft_living y price para realizar la regresión lineal
X = data[:, 5].reshape(-1, 1)
#Xr = np.hstack(np.ones(X.shape(1),1)),X

Y = data[:, 2].reshape(-1, 1)

#X = np.random.rand(())

plt.plot(X, Y, 'ro')
plt.savefig('my_plot.png')

