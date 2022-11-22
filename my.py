data = [1, 2, 3, 4, 8, 9, 3, 6]
WINDOW = 2
FORECAST = 3
for i in range(0, len(data), 1):
    x_i = data[i:i + WINDOW]
    y_i = data[i + WINDOW + FORECAST]
    print(x_i, y_i)

