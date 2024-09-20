import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

# Data used for training and test generation
clean_data = {
    "plains": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
    ],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]        
    ]
}


# View grid based on density
def view_sensor_image(data):
    figure = plt.figure(figsize=(4,4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype='uint8')
    pixels = pixels.reshape((4, 4))
    axes.set_title( "Left Camera:" + data[-1])
    axes.imshow(pixels, cmap='gray')
    plt.show()

# Add noise to data
def blur( data):
    def apply_noise( value):
        if value < 0.5:
            v = random.gauss(0.30, 0.07) # (0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.70, 0.07) # (0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v
    noisy_readings = [apply_noise( v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]

# Generate data based on given input size and data type
def generate_data( data, n, key_label):
    labels = list(data.keys())
    labels.remove(key_label)

    total_labels = len(labels)
    result = []
    # create n "not label" and code as y=0
    count = 1
    while count <= n:
        label = labels[count % total_labels]
        datum = blur(random.choice(data[label]))
        tot_data = datum[0:-1]
        tot_data.append(0)
        result.append(tot_data)
        count += 1

    # create n "label" and code as y=1
    for _ in range(n):
        datum = blur(random.choice(data[key_label]))
        tot_data = datum[0:-1]
        tot_data.append(1)
        result.append(tot_data)
    random.shuffle(result)
    return result

# Create a copy of the data and separate x-values and labels
def separated_data(data):
    y_values = []
    copy_data = copy.deepcopy(data)
    
    for i in range(len(data)):
        copy_data[i].insert(0, 1.0)
        y_val = copy_data[i].pop(-1)
        y_values.append(y_val)

    return (copy_data, y_values)

# Calculate predicted values based on input data
def calculate_yhats(thetas, x_values):
    y_hats = []
   
    for i in range(len(x_values)):
        temp_eval = 0.0

        for j in range(len(x_values[i])):
            temp_eval += x_values[i][j] * thetas[j]

        y_hat = 1 / (1+np.exp(-1*temp_eval))
        y_hats.append(y_hat)
                     
    return y_hats

# Calculate gradient of data
def derivative(j, all_thetas, data):
    total_sum = 0.0
    x_vals, y_vals = separated_data(data)
    y_hats = calculate_yhats(all_thetas, x_vals)

    for i in range(len(y_vals)):
        total_sum += (y_hats[i]-y_vals[i]) * x_vals[i][j]

        derivative_val = total_sum/len(x_vals)
    return derivative_val

# Calculate error of predicted values
def calculate_error(thetas, data):
    total_sum = 0.0
    x_vals, y_vals = separated_data(data)
    y_hats = calculate_yhats(thetas, x_vals)
    for i in range(len(y_vals)):
        total_sum += y_vals[i]*np.log(y_hats[i]) + (1-y_vals[i])*np.log(1-y_hats[i])

    avg = (-1/len(y_vals)) * total_sum
    return avg

# Train model using logarithmic regression
def learn_model(data, verbose=False):
    thetas = [random.uniform(-1, 1) for i in range(len(data[0]))]
    previous_error = 0.0
    current_error = calculate_error(thetas, data)
    iteration = 0
    epsilon = 1e-5
    alpha = 0.1

    while abs(current_error-previous_error) >= epsilon:
        iteration += 1
        new_thetas = []
        for j in range(len(thetas)):
            new_theta= thetas[j] - alpha*derivative(j, thetas, data)
            new_thetas.append(new_theta)
            
        thetas = new_thetas
        previous_error = current_error
        current_error = calculate_error(thetas, data)

        if current_error > previous_error:
            alpha = alpha/10.0

        if verbose and iteration % 1000 == 0:
            print(f"-----------Iteration: {iteration} Current Error: {current_error}-----------")
    
    return new_thetas

# Use model to predict data values
def apply_model(model, test_data, labeled=False):
    model = learn_model(test_data, True)
    x_vals, y_vals = separated_data(test_data)
    predicted_ys = calculate_yhats(model, x_vals)
    results = []
    
    threshold = 0.50
    for i in range(len(y_vals)):
        #print(i,y_vals[i], predicted_ys[i])
        classified_yhats = 0
        if predicted_ys[i] > threshold:
            classified_yhats = 1
        results.append((y_vals[i], classified_yhats))
    
    return results

# Plot confusion matrix of data
def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()

# Calculate confusion matrix and error rate of model
def evaluate(results):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for i in range(len(results)):
        actual_y, predicted_y = results[i]
        
        if actual_y == predicted_y and actual_y == 1:
            true_pos += 1
        if actual_y != predicted_y and actual_y == 0:
            false_pos +=1
        if actual_y == predicted_y and actual_y == 0:
            true_neg += 1
        if actual_y != predicted_y and actual_y == 1:
            false_neg += 1
            
    error_rate = (false_neg+false_pos) / len(results) * 100
    confusion_matrix = [[true_pos, false_pos], [false_neg, true_neg]]
    
    print("Error rate: ", error_rate, "%")
    plot_confusion_matrix(confusion_matrix)
    return error_rate
