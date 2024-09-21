import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

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



def view_sensor_image(data):
    '''
    Plot image based on data
    '''
    
    figure = plt.figure(figsize=(4,4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype='uint8')
    pixels = pixels.reshape((4, 4))
    axes.set_title( "Left Camera:" + data[-1])
    axes.imshow(pixels, cmap='gray')
    plt.show()


def blur( data):
    '''
    Add gaussian noise to data
    '''
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
    '''
    Generate noisy data based on input size n for data icon key_label
    '''
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


def separated_data(data):
    '''
    Separate data into x inputs and y labels
    '''
    y_values = []
    copy_data = copy.deepcopy(data)
    
    for i in range(len(data)):
        copy_data[i].insert(0, 1.0)
        y_val = copy_data[i].pop(-1)
        y_values.append(y_val)

    return (copy_data, y_values)


def calculate_yhats(thetas, x_values):
    '''
    Calculate predicted values based on x input data and thetas
    '''
    y_hats = []
   
    for i in range(len(x_values)):
        temp_eval = 0.0

        for j in range(len(x_values[i])):
            temp_eval += x_values[i][j] * thetas[j]

        y_hat = 1 / (1+np.exp(-1*temp_eval))
        y_hats.append(y_hat)
                     
    return y_hats


def derivative(j, all_thetas, data):
    '''
    Calculate data gradient
    '''
    total_sum = 0.0
    x_vals, y_vals = separated_data(data)
    y_hats = calculate_yhats(all_thetas, x_vals)

    for i in range(len(y_vals)):
        total_sum += (y_hats[i]-y_vals[i]) * x_vals[i][j]

        derivative_val = total_sum/len(x_vals)
    return derivative_val


def calculate_error(thetas, data):
    '''
    Calculate error from predicted y output and actual output
    '''
    total_sum = 0.0
    x_vals, y_vals = separated_data(data)
    y_hats = calculate_yhats(thetas, x_vals)
    for i in range(len(y_vals)):
        total_sum += y_vals[i]*np.log(y_hats[i]) + (1-y_vals[i])*np.log(1-y_hats[i])

    avg = (-1/len(y_vals)) * total_sum
    return avg


def learn_model(data, verbose=False):
    '''
    Train model using logarithmic regression
    '''
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


def apply_model(model, test_data, labeled=False):
    '''
    Use model to predict data values on unseen dataset
    '''
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


def plot_confusion_matrix(confusion_matrix):
    '''
    Display confusion matrix of data
    '''
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()


def evaluate(results):
    '''
    Calculate confusion matrix and error rate of model
    '''
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    # Track predicted value with actual value
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








#--------------------------------------------------------------------------
# Actual streamlit processes


# Sanity check for streamlit version
if st.__version__ != '1.29.0':
    st.warning(f"Warning: Streamlit version is {st.__version__}")

# Show the page title and description.
st.title("Programming Assignment 4")
st.header(
    """
    Terrain Classifier
    """
)

# Create initial grid height and width
if 'world_width' not in st.session_state:
    st.session_state.world_width = 4
if 'world_height' not in st.session_state:
    st.session_state.world_height = 4

# Initialize dataframe when starting up page using initial grid height and width
# Populate with random data
if 'dataframe' not in st.session_state:
    init_data = np.random.rand(st.session_state.world_height, st.session_state.world_width)
    df = pd.DataFrame(init_data, columns=[f"{i}" for i in range(st.session_state.world_width)])
    st.session_state.dataframe = np.round(df, decimals=2)
    st.session_state.init_data = init_data

with st.sidebar:
    container = st.container(border=True)   #Unify all values in sidebar
    container.title("World Size")

    container.write("Select a Width:")
    st.session_state.world_width = container.number_input("Select a Width", min_value=2, max_value=10, value=4, step=1, key="select_width", label_visibility="collapsed")

    container.write("Select a Height:")
    st.session_state.world_height = container.number_input("Select a Height", min_value=2, max_value=10, value=4, step=1, key="select_height", label_visibility="collapsed")

    container.write("#")
    st.session_state.world_terrain = container.selectbox("Terrain: ", ('Plains', 'Forest', 'Hills', 'Swamp'))
            
    submit = container.button("Submit", key="submit_button")

# Display randomized data based on user input for table height and width
if submit:
    init_data = np.random.rand(st.session_state.world_height, st.session_state.world_width)
    df = pd.DataFrame(init_data, columns=[f"{i}" for i in range(st.session_state.world_width)])
    st.session_state.dataframe = np.round(df, decimals=2)
    st.session_state.init_data = init_data

# Make data editable and reflect on display
st.session_state.dataframe = st.data_editor(st.session_state.dataframe)
st.session_state.init_data = st.session_state.dataframe.values

display = st.button("Display")

if display:

    init_data = st.session_state.get('init_data')

    # Plotting based off module 2
    if init_data is not None:

        figure = plt.figure(figsize = (4, 4))
        axes = figure.add_subplot(1, 1, 1)

        pixels = np.array([255 - p * 255 for p in init_data], dtype='uint8')
        pixels = pixels.reshape((st.session_state.world_height, st.session_state.world_width))

        axes.set_title( "Camera View")
        axes.imshow(pixels, cmap='gray')

        axes.set_xticks(np.arange(0, st.session_state.world_width, 2))
        axes.set_yticks(np.arange(0, st.session_state.world_height, 2))

        st.pyplot(figure)

    # Debugging check if data processed
    else:
        st.error("Please press submit before attempting to display data.")
















