#imports
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci  

# get number of vehicles on given lanes
def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


# get total amount of waiting time for given lanes
def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

# ser phase and duration
def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

# model used for predictions in q learning algorithm
class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

# gnn model used for learning node embeddings
class GNN(nn.Module):
    def __init__(self, number_of_node_features, hidden_layers_dims, output_layer_dim):
        super(GNN, self).__init__()
        self.hidden_layers_dims = hidden_layers_dims
        
        self.conv1 = GCNConv(number_of_node_features, hidden_layers_dims[0])
        self.conv2 = GCNConv(hidden_layers_dims[0], hidden_layers_dims[1])
        self.conv3 = GCNConv(hidden_layers_dims[1], output_layer_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, node_features, edges):
        edge_matrix = torch.tensor(edges, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float).to(self.device)

        # split the adj matrix into edge_index and edge_weight in order to pass them to the layers
        rows, cols = edge_matrix.nonzero(as_tuple=True)
        edge_index = torch.stack([rows, cols], dim=0).to(self.device)
        edge_weight = edge_matrix[rows, cols].to(self.device)

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x

# q learning agent
class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )

        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }

    def store_transition(self, state, state_, action,reward, done, junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_.detach().cpu().numpy()
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def reset(self, junction_numbers):
        for junction_number in junctions:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name):
        torch.save(self.Q_eval.state_dict(), f"models/{model_name}.bin")

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch= np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )

        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)

        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )

    def train_gnn(self, num_episodes = 50, num_steps = 500):
        # start traci env
        traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
        )

        # store junction ids, here we are taking only the junctions that have traffic lights
        self.tls_junctions = traci.trafficlight.getIDList()

        # dict for features of junctions
        junction_features_dict = dict()

        # map node id to graph index
        node_id_to_index = {node_id: index for index, node_id in enumerate(self.tls_junctions)}
        
        # weighted adjacency matrix, it will store the distances between traffic lights 
        self.junctions_connections = np.zeros((len(node_id_to_index), len(node_id_to_index)), dtype=np.float32)

        # store distances between traffic lights, useful for creating pairs
        self.junctions_connections_tuple = []

        # iterate through junctions
        for i in range(len(self.tls_junctions)):
          for j in range(i + 1, len(self.tls_junctions)):
            junction_id_i = self.tls_junctions[i]
            junction_id_j = self.tls_junctions[j]
            
            # get coordinates of each junction
            pos_i = traci.junction.getPosition(junction_id_i)
            pos_j = traci.junction.getPosition(junction_id_j)

            # calculate distance between them
            distance = traci.simulation.getDistance2D(pos_i[0], pos_i[1], pos_j[0], pos_j[1])

            # add distance to adj matrix
            self.junctions_connections[node_id_to_index[junction_id_i]][node_id_to_index[junction_id_j]] = distance
            self.junctions_connections[node_id_to_index[junction_id_j]][node_id_to_index[junction_id_i]] = distance

            # add edge and distance to edges tuple array
            self.junctions_connections_tuple.append((self.tls_junctions[i], self.tls_junctions[j], distance))

        # sort array of edges tuple array
        self.junctions_connections_tuple = sorted(self.junctions_connections_tuple, key=lambda edge: edge[2])

        # gnn params
        number_of_node_features = 4
        hidden_layers_dims = [256, 256]
        output_layer_dim = 128

        # initialize gnn
        self.gnn = GNN(number_of_node_features, hidden_layers_dims, output_layer_dim)

        # optimizer
        optimizer = optim.Adam(self.gnn.parameters(), lr=0.1)

        # loss function
        contrastive_loss = nn.MarginRankingLoss(margin=0.5)

        traci.close()

        for epoch in range(num_episodes):
            traci.start(
                [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
            )

            for step in range(num_steps):
                # run a simulation to extract data for input of gnn
                traci.simulationStep()

                #get junction features, i.e. number of cars on each controlled lane
                for junction_id in self.tls_junctions:
                    controled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    
                    junction_features_dict[junction_id] = torch.tensor(list(vehicles_per_lane.values()))

                optimizer.zero_grad()
          
                node_features = torch.stack([junction_features_dict[junction_id] for junction_id in self.tls_junctions], dim=0).to(self.gnn.device)

                # forward pass
                embeddings = self.gnn(node_features, self.junctions_connections)

                #generate positive and negative node pairs
                positive_pairs = []
                negative_pairs = []

                # the positive pairs are represented by the first half of the sorted distances,
                # meaning the traffic lights closest to each other represent the positive pairs
                # and the negative pairs are represented by the traffic lights that are the furthest away
                # from each other 
                for index, edge_tuple in enumerate(self.junctions_connections_tuple):
                    junction_id_1 = edge_tuple[0]
                    junction_id_2 = edge_tuple[1]

                    if index < len(self.junctions_connections_tuple) / 2:
                        positive_pairs.append((embeddings[node_id_to_index[junction_id_1]], embeddings[node_id_to_index[junction_id_2]]))
                    else:
                        negative_pairs.append((embeddings[node_id_to_index[junction_id_1]], embeddings[node_id_to_index[junction_id_2]]))

                # convert the pairs to tensors
                tensor_list = [torch.stack(t) for t in positive_pairs]
                positive_pairs = torch.stack(tensor_list)

                tensor_list = [torch.stack(t) for t in negative_pairs]
                negative_pairs = torch.stack(tensor_list)

                # compute similarity scores
                positive_scores = torch.cosine_similarity(positive_pairs[:, 0], positive_pairs[:, 1], dim=-1)
                negative_scores = torch.cosine_similarity(negative_pairs[:, 0], negative_pairs[:, 1], dim=-1)

                # generate target labels
                target = torch.ones_like(positive_scores)

                # compute the contrastive loss
                loss = contrastive_loss(positive_scores, negative_scores, target)

                # backward prop
                loss.backward()

                # weight update
                optimizer.step()
                
                if step == num_steps - 1:
                    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        
            traci.close()


def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
    """execute the TraCI control loop"""
    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    )

    # get junctions with traffic lights
    tls_junctions = traci.trafficlight.getIDList()

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=128,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=2,
        junctions=tls_junctions,
    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))

    print(brain.Q_eval.device)

    traci.close()

    brain.train_gnn()
    
    for e in range(epochs):
        # start traci
        if train:
            traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
            [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"epoch: {e}")

        # junction options
        select_lane = [
            ["yyyrrryyyrrr", "GGgrrrGGgrrr"],
            ["rrryyyrrryyy", "rrrGGgrrrGGg"],
        ]

        step = 0
        total_time = 0
        min_duration = 5
        
        traffic_lights_time = dict()
        prev_embedding = dict()
        prev_wait_time = dict()
        prev_action = dict()

        # initialize input data for gnn
        junction_features_dict = dict()

        for junction_id in brain.tls_junctions:
            prev_wait_time[junction_id] = 0
            prev_action[junction_id] = 0
            prev_embedding[junction_id] = 0
            traffic_lights_time[junction_id] = 0

        while step <= steps:
            # simulate traffic step
            traci.simulationStep()

            waiting_time = dict()

            #get junction features
            for junction_id in brain.tls_junctions:
                controled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                
                junction_features_dict[junction_id] = torch.tensor(list(vehicles_per_lane.values()))

            node_ids = list(junction_features_dict.keys())
            node_features = torch.stack([junction_features_dict[node_id] for node_id in node_ids], dim=0)

            # map node id to embedding index
            node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

            # forward pass
            embeddings = brain.gnn(node_features, brain.junctions_connections)
            
            for junction_id in brain.tls_junctions:
                # calculate waiting time
                controled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                waiting_time[junction_id] = get_waiting_time(controled_lanes)
                total_time += waiting_time[junction_id]

                # take action only if the traffic light has finished its current phase
                if traffic_lights_time[junction_id] == 0:
                    #reward received
                    reward = -1 *  waiting_time[junction_id]

                    # get embedding of current traffic light
                    node_index = node_id_to_index[junction_id]
                    state_ = embeddings[node_index]

                    # get last embedding of this traffic light
                    state = prev_embedding[junction_id]

                    # store current embedding
                    prev_embedding[node_index] = state_ 

                    # store transition
                    brain.store_transition(state, state_, prev_action[junction_id], reward, (step==steps), junction_id)

                    # selecting new action based on current state
                    phase = brain.choose_action(state_)
                    prev_action[junction_id] = phase

                    # set phase duration
                    phaseDuration(junction_id, 6, select_lane[phase][0])
                    phaseDuration(junction_id, min_duration + 10, select_lane[phase][1])

                    # set time of junction
                    traffic_lights_time[junction_id] = min_duration + 10
                    if train:
                        brain.learn(junction_id)
                else:
                    traffic_lights_time[junction_id] -= 1
            step += 1

        print("total_time",total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)
