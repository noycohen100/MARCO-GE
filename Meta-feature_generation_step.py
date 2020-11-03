import dgl
import os
import torch
from operator import itemgetter
import networkx as nx
from torch.utils.data import DataLoader
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# index ='average'
# label_file_name='training/graphs_labels' +index +'.txt'

batch_size = 20
node_embedding_dim = 64
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, embedding):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h=g.in_degrees()
        h = g.in_degrees().view(-1, 1).float()
        h.data = embedding
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg), hg

#Generating dictionary of labels: <dataset_name, label(best algorithm)>
def get_labels(labels_path):
    f = open(labels_path)
    line = f.readline()
    dict_labels={}
    while line:
        splits=line.split(";")
        dataset_name = splits[0]
        dataset_label=int(splits[1].split("\n")[0])
        dict_labels[dataset_name]=dataset_label
        line= f.readline()
    f.close()
    return dict_labels

#Retrive nodes' embedding
def get_node_embedding(embedding_dir):
    print('get graph from ' + embedding_dir)
    num_embeddings = sum([len(files) for root, dirs, files in os.walk(embedding_dir)])
    graphs = []
    graphs_dic={}
    for file in sorted(os.listdir(embedding_dir), key=lambda s: s.lower()):
        f = open(embedding_dir+"/"+ file, "r")
        line = f.readline()
        head = line.strip().split(" ")
        line = f.readline()
        vector_dict = {}
        while line:
            curline = line.strip().split(" ")
            vector_dict[int(curline[0])] = [float(s) for s in curline[1:]]
            line = f.readline()
        f.close()

        vector_dict = dict(sorted(vector_dict.items(), key=lambda e: e[0]))
        graphs.append(np.array(list(vector_dict.values())))
        graphs_dic[file.split(".")[0]]=np.array(list(vector_dict.values()))
 
    return graphs_dic


def collate(samples):
    graphs, labels, embedding= map(list,zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), embedding




def retrive_graphs(dict_nodes_embedding, EdgeLists_folder, datasets_folder):
    #Graph_list = mp()
    Graph_list=[]
    for file in sorted(os.listdir(datasets_folder),key=lambda s: s.lower()):
        file = file.split('\\')[-1].split('.')[0]
        G = nx.read_edgelist(EdgeLists_folder+file+".file", create_using=nx.Graph, nodetype=int, data=(('weight',float),))
        G = nx.Graph(G)
        Graph_list.append((file.split(".")[0], G))
    datasets = []
    # dict_labels = get_labels("training/graphs_labels"+index+".txt")
    dict_labels = get_labels(label_file_name)
    for (dataset_name,graph) in Graph_list:
        c = dgl.DGLGraph()
        c.from_networkx(graph, edge_attrs=['weight'])
        c.set_n_initializer(dgl.init.zero_initializer)
        datasets.append((c, dict_labels[dataset_name],torch.FloatTensor(dict_nodes_embedding[dataset_name])))

    return datasets


def initiate_feat(embedding):
    embedding_array=[]
    if len(embedding) == 1:
        embedding_array = torch.Tensor(embedding[0])
    else:
        for j in range(1, len(embedding)):
            if j == 1:
                embedding_array = torch.FloatTensor(np.concatenate((embedding[j - 1], embedding[j])))
            else:
                embedding_array = torch.FloatTensor(np.concatenate((embedding_array, embedding[j])))
    
    return embedding_array



def graphClassification(datasets_folder, EdgeLists_folder, NodesEmbedding_folder,number_of_epochs, embedding_size, num_classes,clustering_index, labels_file):
    files_name= [file for file in sorted(os.listdir(datasets_folder),key=lambda s: s.lower())]
    print(len(files_name))
    dict_nodes_embedding = get_node_embedding(NodesEmbedding_folder)
    graphs= retrive_graphs(dict_nodes_embedding, EdgeLists_folder, datasets_folder)
    loo = LeaveOneOut()
    splits = loo.split(graphs)

    for train_index, test_index in splits:
        # train_index = np.insert(train_index, 0,test_index[0])
        train_set=itemgetter(*train_index)(graphs)
        test_set=[itemgetter(*test_index)(graphs)]

        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    # Create model
        for i in range(0, batch_size-1):
            c = dgl.DGLGraph()
            c.add_nodes(1)
            test_set.append((c,0,torch.FloatTensor(torch.zeros(1,64))))
        model = Classifier(node_embedding_dim, embedding_size, num_classes)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.006)
        model.train()
        epoch_losses = []
        
        for epoch in range(number_of_epochs):
            epoch_loss = 0
            counter =0
            
            for iter, (bg, label, embedding) in enumerate(data_loader):
                embedding_array=[]
                embedding_array=initiate_feat(embedding)
                prediction,hg = model(bg, embedding_array)
                
                if epoch == number_of_epochs -1:
                    embedding_dir = "Embeddings"
                    os.mkdir(embedding_dir)
                    for hidden in hg.detach().numpy():
                        dataset_hidden_name=files_name[train_index[counter]].split(".")[0]
                        hidden.tofile(embedding_dir+"/"+ dataset_hidden_name + ".csv",sep=',')
                        counter = counter + 1

                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
        model.eval()
            # Convert a list of tuples to two lists
        test_X, test_Y,embedding_test = map(list, zip(*test_set))
        test_bg = dgl.batch(test_X)
        true_label=test_Y[0]
        dataset_test_name =files_name[test_index[0]].split(".")[0]
        print(dataset_test_name)
        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        probs_Y,hidden_layer = model(test_bg,initiate_feat(embedding_test))
        probs_Y = torch.softmax(probs_Y, 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
        break
