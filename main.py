from Graph_representation_step import Graph_Representation
from Clustering_algorithm_evaluation_step import Clustering_algorithm_evaluation
from Meta_feature_generation_step import graphClassification
from Meta_model_creation_step import model
import merge_results
import argparse
import os
import json


EdgeLists_folder = "EdgeLists"
NodesEmbedding_folder = "nodeEmbeddings"
N_clustering_algs =13

def parse_args(parser):
    parser.add_argument('--Df', nargs='?', default='Datasets', help='Input datasets folder')
    parser.add_argument('--em', default='average', help="Select clustering measure for evaluating the clustering algorithms: average/all")
    parser.add_argument('--mm', default='average', help="Select clustering index for training the meta-model: average/DB/DU/CH/SIL/MC/DB/HKK/HL/Scat/Xie")
    parser.add_argument('--ne',type =int ,default=50, help='GCNN model - Number of epochs')
    parser.add_argument('--es', type = int, default=300, help='GCNN model - Embedding size')
    return parser.parse_args()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MARCO-GE")
    args = parse_args(parser)
    Graph_Representation(args.Df, EdgeLists_folder, NodesEmbedding_folder)
    print(args.Df)
    Clustering_algorithm_evaluation(args.Df, args.em)
    label_file_name = merge_results.find_best_algorithm(args.mm)
    embedding_dir = graphClassification(args.Df, EdgeLists_folder+"/", NodesEmbedding_folder+"/",args.ne, args.es, N_clustering_algs, args.mm,label_file_name)

    with open('XGBoostHyperParameters/'+args.mm +".json", "r") as fj:
        parameters = json.load(fj)
    training_file_name = merge_results.union_algorithms_performance(args.mm)
    model(embedding_dir, parameters['depth'],parameters['trees'], parameters['lr'], parameters['col_s'], parameters['sub_s'], parameters['min_c'],parameters['seed'] ,args.es,training_file_name, N_clustering_algs, args.mm )
