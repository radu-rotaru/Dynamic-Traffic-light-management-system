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

    def forward(self, node_features, edges):
        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv3(x, edge_index)
        return x

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
        # print(f"state_: {state_}")
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_.detach()
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
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

    def train_gnn(self, num_episodes = 100, num_steps = 500):
        traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
        )

        # store traffic light ids, these are the junctions where we have agents
        self.tls_junctions = traci.trafficlight.getIDList()

        # dict for features of junctions
        junction_features_dict = dict()

        self.junctions_connections = []

        # read network file
        net = sumolib.net.readNet('/home/radurot/licenta/framework3/Dynamic-Traffic-light-management-system/maps/city1.net.xml')
        self.gnn_nodes = [node.getID() for node in net.getNodes()]

        # map node id to graph index
        node_id_to_index = {node_id: index for index, node_id in enumerate(self.gnn_nodes)}

        self.gnn_edges = net.getEdges()

        edges = net.getEdges()

        for edge in edges:
            toNode = node_id_to_index[net.getEdge(edge.getID()).getToNode().getID()]
            fromNode = node_id_to_index[net.getEdge(edge.getID()).getFromNode().getID()]
            self.junctions_connections.append((fromNode, toNode))

        # gnn params
        number_of_node_features = 2
        hidden_layers_dims = [128, 128]
        output_layer_dim = 1

        self.gnn = GNN(number_of_node_features, hidden_layers_dims, output_layer_dim)

        # optimizer
        optimizer = optim.Adam(self.gnn.parameters(), lr=0.01)

        # loss function
        contrastive_loss = nn.MarginRankingLoss(margin=0.5)

        for epoch in range(num_episodes):
            # run a simulation to take and extract data for input of gnn
            traci.simulationStep()

            #get junction features
            for junction_id in self.gnn_nodes:
                cars_at_junction = 0
                phase = -1
                
                if junction_id in self.tls_junctions:
                    phase = traci.trafficlight.getPhase(junction_id)
                    for lane in traci.trafficlight.getControlledLanes(junction_id):
                        cars_at_junction += traci.lane.getLastStepVehicleNumber(lane)

                junction_features_dict[junction_id] = torch.tensor([phase, cars_at_junction])

            optimizer.zero_grad()
            
            node_ids = list(junction_features_dict.keys())
            node_features = torch.stack([junction_features_dict[node_id] for node_id in node_ids], dim=0)

            # print(f"Node ids: {junction_features_dict.keys()}")

            # print(f"Node features: {node_features}")

            # forward pass
            embeddings = self.gnn(node_features, self.junctions_connections)

            #generate positive and negative node pairs
            positive_pairs = []
            for edge in self.junctions_connections:
                positive_pairs.append((embeddings[edge[0]], embeddings[edge[1]]))
            
            negative_pairs = []
            for i in range(len(self.gnn_nodes)):
                if len(negative_pairs) == len(positive_pairs):
                    break
                for j in range(i + 1, len(self.gnn_nodes)):
                    pair = (embeddings[i], embeddings[j])
                    if pair not in positive_pairs:
                        negative_pairs.append(pair)
                    if len(negative_pairs) == len(positive_pairs):
                        break

            tensor_list = [torch.stack(t) for t in positive_pairs]
            positive_pairs = torch.stack(tensor_list)

            tensor_list = [torch.stack(t) for t in negative_pairs]
            negative_pairs = torch.stack(tensor_list)
            # compute similarity scores
            similarity_scores = torch.cosine_similarity(positive_pairs, negative_pairs, dim=2)

            # generate target labels
            target = torch.ones_like(similarity_scores)

            # compute the contrastive loss
            loss = contrastive_loss(similarity_scores, target, torch.ones_like(target))

            # backward prop
            loss.backward()

            # weight update
            optimizer.step()

            # print the loss
            if epoch % 10 == 0:
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

    tls_junctions = traci.trafficlight.getIDList()

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims=1,
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
            traci.simulationStep()

            waiting_time = dict()

            #get junction features
            for junction_id in brain.gnn_nodes:
                cars_at_junction = 0
                phase = -1
                
                if junction_id in brain.tls_junctions:
                    phase = traci.trafficlight.getPhase(junction_id)
                    for lane in traci.trafficlight.getControlledLanes(junction_id):
                        cars_at_junction += traci.lane.getLastStepVehicleNumber(lane)

                junction_features_dict[junction_id] = torch.tensor([phase, cars_at_junction])

            node_ids = list(junction_features_dict.keys())
            node_features = torch.stack([junction_features_dict[node_id] for node_id in node_ids], dim=0)

            # map node id to embedding index
            node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

            # forward pass
            embeddings = brain.gnn(node_features, brain.junctions_connections)
            
            for junction_id in brain.tls_junctions:
                controled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                waiting_time[junction_id] = get_waiting_time(controled_lanes)
                total_time += waiting_time[junction_id]

                if traffic_lights_time[junction_id] == 0:
                    #reward received
                    reward = -1 *  waiting_time[junction_id]

                    # change with embedding
                    node_index = node_id_to_index[junction_id]
                    state_ = embeddings[node_index]

                    # change with embedding
                    state = prev_embedding[junction_id]

                    # should store embedding
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
