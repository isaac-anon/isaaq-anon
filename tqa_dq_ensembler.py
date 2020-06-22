from transformers import RobertaTokenizer
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse

from aux_methods import get_data_ndq, process_data_ndq, get_data_dq, validation_ndq, validation_dq, get_upper_bound, superensembler

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainingslist', default=["checkpoints/tmc_dq_roberta_IR_e4.pth", "checkpoints/tmc_dq_roberta_NSP_e4.pth", "checkpoints/tmc_dq_roberta_NN_e2.pth", "checkpoints/dmc_dq_roberta_IR_e4.pth", "checkpoints/dmc_dq_roberta_NSP_e2.pth", "checkpoints/dmc_dq_roberta_NN_e4.pth"], help='list of paths of the pretrainings model. They must be three. Default: checkpoints/tmc_dq_roberta_IR_e4.pth, checkpoints/tmc_dq_roberta_NSP_e4.pth, checkpoints/tmc_dq_roberta_NN_e2.pth, checkpoints/dmc_dq_roberta_IR_e4.pth, checkpoints/dmc_dq_roberta_NSP_e2.pth, checkpoints/dmc_dq_roberta_NN_e4.pth')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default= 32, type=int, help='size of the batches. Default: 512')
    args = parser.parse_args()
    print(args)
    
    models = [torch.load(args.pretrainingslist[0]), torch.load(args.pretrainingslist[1]), torch.load(args.pretrainingslist[2]), torch.load(args.pretrainingslist[3]), torch.load(args.pretrainingslist[4]), torch.load(args.pretrainingslist[5])]
    retrieval_solvers = ["IR", "NSP", "NN", "IR", "NSP", "NN"]
    model_types = ["tmc", "tmc", "tmc", "dmc", "dmc", "dmc"]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    max_len = args.maxlen
    batch_size = args.batchsize
    
    feats_train = []
    feats_test = []
    for model, model_type, retrieval_solver in zip(models, model_types, retrieval_solvers):
        if args.device=="gpu":
            device = torch.device("cuda")
            model.cuda()
        if args.device=="cpu":
            device = torch.device("cpu") 
            model.cpu()
        model.eval()
        print("\n")
        print(retrieval_solver)
        if model_type == "dmc":
            print("val")
            raw_data_train = get_data_dq("val", retrieval_solver, tokenizer, max_len)
            feats_train.append(validation_dq(model, raw_data_train, batch_size, device))
            labels_train = raw_data_train[-1]
            print("test")
            raw_data_test = get_data_dq("test", retrieval_solver, tokenizer, max_len)
            feats_test.append(validation_dq(model, raw_data_test, batch_size, device))
            labels_test = raw_data_test[-1]
        if model_type == "tmc":
            print("val")
            raw_data_train = get_data_ndq("dq", "val", retrieval_solver, tokenizer, max_len)
            train_dataloader = process_data_ndq(raw_data_train, batch_size, "val")
            feats_train.append(validation_ndq(model, train_dataloader, device))
            labels_train = raw_data_train[-1]
            print("test")
            raw_data_test = get_data_ndq("dq", "test", retrieval_solver, tokenizer, max_len)
            test_dataloader = process_data_ndq(raw_data_test, batch_size, "test")
            feats_test.append(validation_ndq(model, test_dataloader, device))
            labels_test = raw_data_test[-1]
   
    upper_bound_train = get_upper_bound(feats_train, labels_train)
    res = superensembler(feats_train, feats_test, labels_train, labels_test)
    print("\nFINAL RESULTS:")
    print("TEST SET: ")
    print(res)

    res = superensembler(feats_test, feats_train, labels_test, labels_train)
    print("VALIDATION SET: ")
    print(res)

if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])