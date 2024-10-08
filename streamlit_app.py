import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import random
import pandas as pd

from IPython.display import display_html
from typing import List, Tuple, Dict, Callable
from copy import deepcopy


def generate_random_world(world_height, world_width, available_terrain):
    """
    Randomly generate a world based on user input height and width

    Parameters:
    world_height: height of world
    wold_width: width of world
    available_terrain: values to choose from for terrain

    Returns:
    world: the randomly generated world
    """
    world = []
    terrain_keys = list(available_terrain.keys())
    
    for i in range(world_height):
        world_row = []
        for j in range(world_width):
            
            current_choice = random.choice(terrain_keys)
            world_row.append(current_choice)
            
        world.append(world_row)

    return world

def display_emoji_grid(emoji_grid):
    """
    Display a List of Lists of emojis in a perfect grid (table)
    
    Parameters:
    emoji_grid (list of list of str): A 2D list containing emojis to display in a grid.
    """
    # Create HTML table
    html = '<table style="border-collapse: collapse;">'
    
    for row in emoji_grid:
        html += '<tr>'
        for emoji in row:
            html += f'<td style="border: none; padding: 0px; text-align: center; font-size: 1em;">{emoji}</td>'
        html += '</tr>'
    
    html += '</table>'
    
    return html

MOVES = [(0,-1), (1,0), (0,1), (-1,0)]
COSTS = { '🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7, '🌋': 1000}


def successor(world: List[List[str]], node: Tuple[int, int], moves: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Finds the next potential moves from the current node

    Parameters:
    world: a 2D list representation of the world
    node: a Tuple representation of world coordinates
    moves: set of valid movement from the current node

    Returns: 
    children: coordinates after valid movement was done
    """
    x, y = node
    children = []

    for move in moves:
        child_x, child_y = x+move[0], y+move[1]
    
        if (0 <= child_x < len(world)) and (0 <= child_y < len(world[0])):
            child_node = (child_x, child_y)
            children.append(child_node)

    return children


def paths(node: Tuple[int, int], parent_mapping: Dict[Tuple[int, int], Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Finds the path from current node to beginning based on parents of each node

    Parameters:
    node: current node being examined
    parent_mapping: dictionary of parent:child mappings of each visited node at the time

    Returns:
    path in reverse format
    """
    path = []
    
    while node is not None:
        path.append(node)
        node = parent_mapping[node]

    return path[::-1]


def movement_path(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Calculates the move made from node to node based on the set of valid moves

    Parameters:
    path: the traversed path in the world
    goal: the position of the goal node in (x, y) format

    Returns: 
    move_path: the necessary movement values to reach the goal node from the start node
    """
    move_path = []
    
    for i in range(len(path)):
        if path[i] == goal:
            return move_path
            
        x, y = path[i]
        new_x, new_y = path[i+1]

        movement = (new_x-x, new_y-y)
        move_path.append(movement)

    return move_path

def heuristic(current_node: Tuple[int, int], goal: Tuple[int, int], current_cost: int, current_node_cost: int) -> int: 
    """
    Calculate next shortest path based on the current cost up to the current node and the remaining potential minimum cost to the goal node (manhattan distance)

    Parameters:
    current_node: current location of the robot in (x, y) coordinates
    goal: the position of the goal node in (x, y) coordinates
    current_cost: the cost of the currently explored path
    current_node_cost: the cost of the current node to add in the total cost path

    Returns:
    f_n: the heuristic, cost of A* search, f(n) = g(n) + h(n)
    """
    x, y = current_node
    goal_x, goal_y = goal

    g_n = current_cost + current_node_cost
    h_n = abs((x-goal_x) + (y-goal_y))
    f_n = g_n + h_n

    return f_n


def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    """
    Perform A* search on a given world, finding the best path it can.

    Parameters: 
    world: the actual context for the navigation problem
    start: the starting location of the robot in (x, y) coordinates
    goal: the desired goal position for the bot in (x, y) coordinates
    costs: a dictionary of costs for each type of terrain in the world
    moves: the legal movement model expressed in offset in world
    heuristic: the heuristic function h(n)

    Returns: the offsets needed to get from start state to the goal state
    """    
    frontier = [start]
    parent_explored = {start: None}
    cost_so_far = {start: costs[world[start[0]][start[1]]]}
    path = []

    while frontier:
        current_node = frontier.pop(0)

        if current_node == goal:
            world_path = paths(current_node, parent_explored)
            return world_path
            
        children = successor(world, current_node, moves)
        for child in children:

            child_cost = costs[world[child[0]][child[1]]]
            if child_cost == 1000:
                 continue
            new_cost = heuristic(child, goal, cost_so_far[current_node], child_cost)

            if ~(child in frontier) and (child not in cost_so_far or new_cost < cost_so_far[child]):
                cost_so_far[child] = new_cost
                frontier.append(child)
                parent_explored[child] = current_node

    return None # if path is not found

def pretty_print_helper(x_move, y_move, node, goal) -> str:
    """
    Finds the proper direction emoji given the node's movement

    Parameters:
    x_move: movement in x direction
    y_move: movement in y direction
    node: examined node
    goal: goal position

    Returns: 
    movement_emoji: the expected movement emoji given the movement in x and y direction
    """
    movement_emoji = ''
    
    if x_move == 0 and y_move == -1:
        movement_emoji = '⏪'
    if x_move == 0 and y_move == 1:
        movement_emoji = '⏩'
    if x_move == -1 and y_move == 0:
        movement_emoji = '⏫'
    if x_move == 1 and y_move == 0:
        movement_emoji = '⏬'

    return movement_emoji


def pretty_print_path( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    """
    Overlays the optimal path in the world display from start to goal. Also prints the total movement cost of the path

    Parameters:
    world: the world to be printed upon
    path: the path from start to goal, in offsets
    start: the starting location for the path
    goal: the goal location for the path
    costs: the cost for each action

    Returns: 
    path_cost: the path cost
    """
    x, y = start
    path_cost = 0
    copy_world = deepcopy(world)
    move_path = movement_path(path, goal)
    i = 0
    for node in path:
        movement_emoji = ''
        if (node[0], node[1]) == goal and i == len(move_path):
            movement_emoji = '🎁'
        else:
            x_move, y_move = move_path[i]     
            movement_emoji = pretty_print_helper(x_move, y_move, node, goal)
            
        path_cost += costs[world[node[0]][node[1]]]
        copy_world[node[0]][node[1]] = movement_emoji
        i += 1
        
    traversed_world = display_emoji_grid(copy_world)
    return path_cost, copy_world
#--------------------------------------------------------------------------
# Actual streamlit processes


# Sanity check for streamlit version
if st.__version__ != '1.29.0':
    st.warning(f"Warning: Streamlit version is {st.__version__}")

# Show the page title and description.
st.title("Programming Assignment 4")
st.header(
    """
    World Traversal- A* Search
    """
)
costs_string = {'🌾': '1', '🌲': '3', '⛰': '5', '🐊': '7', '🌋': 'Impassable'}

st.text("The following is a list of possible terrain and their associated movement cost: ")
costs_df = pd.DataFrame(list(costs_string.items()), columns=['Terrain', 'Movement Cost'])
st.table(costs_df)

on = st.toggle("How To Play:")

if on:
    st.write(
        """
        1. Select the size of the world for generation.
        2. Set Start and Goal: Use the sidebar to select the start and goal coordinates from the pathfinding.
        3. Find Path: Click the 'Find Path' button in the sidebar after setting the coordinates.
        4. View Results: The optimal path from start to goal, along with its total cost and visual representation,
           will be displayed below.
        5. The travel cost for each terrain is shown above.
        
        Note: Coordinates will be in (x, y) format: (left-right, up-down)
        """
    )

# Create initial session variables
if 'world_width' not in st.session_state:
    st.session_state.world_width = 7
if 'world_height' not in st.session_state:
    st.session_state.world_height = 7
if 'world' not in st.session_state:
    st.session_state.world = generate_random_world(st.session_state.world_height, st.session_state.world_width, COSTS)
if 'emoji_data' not in st.session_state:
    st.session_state.emoji_data = display_emoji_grid(st.session_state.world)
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = [(j, i) for i in range(st.session_state.world_height) for j in range(st.session_state.world_width)]


# Maintain instance of emoji_data even when updating coordinates
emoji_display_placeholder = st.empty()    
emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)

with st.sidebar:
    container = st.container(border=True)   # Unify all world generation values
    container.title("World Size")

    container.write("Select a Width:")
    st.session_state.world_width = container.number_input("Select a Width", min_value=2, max_value=40, value=7, step=1, key="select_width", label_visibility="collapsed")

    container.write("Select a Height:")
    st.session_state.world_height = container.number_input("Select a Height", min_value=2, max_value=40, value=7, step=1, key="select_height", label_visibility="collapsed")
    
    submit = container.button("Generate World", key="submit_button")

    container2 = st.container(border=True)    # Unify all coordinate values 
    container2.title("Location Objective")
    
    container2.write("Select a Starting Coordinate: ")
    st.session_state.starting_coord = container2.selectbox("Starting Point", st.session_state.coordinates, label_visibility="collapsed")
    
    container2.write("Select a Goal Coordinate: ")
    st.session_state.goal_coord = container2.selectbox("Goal Point", st.session_state.coordinates, label_visibility="collapsed")

    find_path = container2.button("Find Path", key="find_path_button")

# Display randomized data based on user input for table height and width
if submit:
    st.session_state.world = generate_random_world(st.session_state.world_height, st.session_state.world_width, COSTS)
    st.session_state.coordinates = [(j, i) for i in range(st.session_state.world_height) for j in range(st.session_state.world_width)]
    st.session_state.emoji_data = display_emoji_grid(st.session_state.world)
    emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)


if find_path:

    init_data = st.session_state.world
    # Plotting based off module 2
    if init_data is not None:

        reverse_start = st.session_state.starting_coord
        reverse_goal = st.session_state.goal_coord

        # Swap coordinates to program friendly format
        start = (reverse_start[1], reverse_start[0])
        goal = (reverse_goal[1], reverse_goal[0])

        # Check if starting or ending on mountain
        if init_data[reverse_start[1]][reverse_start[0]] == '🌋':
            st.write("Invalid starting location, please pick a new coordinate.")
            st.session_state.emoji_data = display_emoji_grid(init_data)
            emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)
            st.stop()

        if init_data[reverse_goal[1]][reverse_goal[0]] == '🌋':
            st.write("Invalid goal location, please pick a new coordinate.")
            st.session_state.emoji_data = display_emoji_grid(init_data)
            emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)
            st.stop()
        
        world_traversal = a_star_search(init_data, start, goal, COSTS, MOVES, heuristic)
        path_cost = 0

        # Paths greater than 1000 are technically not real- mountains are impassable
        if path_cost >= 1000:
            no_path = st.write("No path was found")
            st.session_state.emoji_data = display_emoji_grid(init_data)
            emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)

        elif world_traversal is None:
            no_path = st.write("No path was found")
            st.session_state.emoji_data = display_emoji_grid(init_data)
            emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)
            
        elif world_traversal is not None:
            path_cost, found_goal_world = pretty_print_path(init_data, world_traversal, start, goal, COSTS)
            st.session_state.emoji_data = display_emoji_grid(found_goal_world)

            st.write(f"Path was found! Total cost is {path_cost}")
            emoji_display_placeholder.markdown(st.session_state.emoji_data, unsafe_allow_html=True)




















