import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=2, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs=[1, 2, 3, 4, 5, 6]
ys=[2.3, 3.8, 6.7, 8.8, 9.2, 13.3]
model.fit(xs, ys, epochs=100)
for i in range(7, 13):
    value = model.predict([i])
    print(f"Prediction for {i}: {value}")
    xs.append(i)
    ys.append(value[0][0])

plt.figure(figsize=(7,8))
plt.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color='orange', label='Line through first and last points')
plt.scatter(xs, ys, color='blue', marker='o', label='Data points')
plt.xlabel('xs')
plt.ylabel('ys')
plt.title('Line through first and last data points')
plt.legend()
plt.grid(True)
plt.show()
