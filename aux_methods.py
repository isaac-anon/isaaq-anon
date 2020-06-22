from transformers import RobertaForSequenceClassification, AdamW, BertConfig, BertTokenizer, BertPreTrainedModel, BertModel, BertForMultipleChoice, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import json
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
import random
from transformers.modeling_utils import PreTrainedModel

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from PIL import Image

import torchvision.models as models

class ResnetRobertaBUTD(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertaBUTD, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        
        self.resnet = models.resnet101(pretrained=True)
        
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000,1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.boxes = torch.nn.Sequential(torch.nn.Linear(4,1024),torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.att = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh(),torch.nn.Linear(1024,1),torch.nn.Softmax(dim=0))
        
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images=None,coords=None,labels=None):
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        
        img_b = []
        coord_b = []
        for img_q, coord_q in zip(images,coords):
            for img_file, coord in zip(img_q,coord_q):
                img_b.append(img_file)
                coord_b.append(coord)
        
        roi_b = []
        for image, coord, roberta_b in zip(img_b,coord_b, out_roberta):
            img_v = get_rois(image, coord[:32])
            coord_v = torch.tensor(coord[:32]).cuda()
            out_boxes = self.boxes(coord_v)
            out_resnet = self.resnet(img_v)
            out_resnet = self.feats(out_resnet)
            out_resnet = self.feats2(out_resnet)
            out_resnet = out_resnet.view(-1,1024)
            out_roi = (out_resnet + out_boxes)/2
            
            n_rois = np.shape(out_roi)[0]
            
            out_att = torch.cat((out_roi, roberta_b.repeat(n_rois,1)),1)
            out_att = self.att(out_att)
            out_att = out_att*out_roi
            out_att = torch.sum(out_att, dim=0)
            roi_b.append(out_att)
        out_visual = torch.stack(roi_b, dim=0)
        
        final_out = out_roberta * out_visual
        
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    #return np.sum(pred_flat == labels_flat) / len(labels_flat)
    return np.sum(pred_flat == labels_flat), np.sum(pred_flat != labels_flat)

def get_rois(img_path, vectors):
    image = Image.open(img_path)
    rois = []
    for vector in vectors:
        roi_image = image.crop(vector)
        roi_image = roi_image.resize((224, 224), Image.ANTIALIAS)
        roi_image = np.array(roi_image)
        roi_image = torch.tensor(roi_image).type(torch.FloatTensor).permute(2,0,1).cuda()
        rois.append(roi_image)
    rois = torch.stack(rois, dim=0)
    return rois

def get_choice_encoded(text, question, answer, max_len, tokenizer):
    if text != "":
        first_part = text
        second_part = question + " " + answer
        encoded = tokenizer.encode_plus(first_part, second_part, max_length=max_len, pad_to_max_length=True)
    else:
        encoded = tokenizer.encode_plus(question + " " + answer, max_length=max_len, pad_to_max_length=True)
    input_ids = encoded["input_ids"]
    att_mask = encoded["attention_mask"]
    return input_ids, att_mask

def get_data_tf(split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    labels_list=[]
    cont = 0
    with open("jsons/tqa_tf.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["sentence_"+retrieval_solver]
        encoded = tokenizer.encode_plus(text, question, max_length=max_len, pad_to_max_length=True)
        input_ids = encoded["input_ids"]
        att_mask = encoded["attention_mask"]
        label = 0
        if doc["correct_answer"] == "true":
            label = 1
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)
        labels_list.append(label)
    return [input_ids_list, att_mask_list, labels_list]

def get_data_ndq(dataset_name, split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    labels_list=[]
    cont = 0
    with open("jsons/tqa_"+dataset_name+".json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_"+retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        if dataset_name == "ndq":
            counter = 7
        if dataset_name == "dq":
            counter = 4
        for count_i in range(counter):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, labels_list]

def process_data_ndq(raw_data, batch_size, split):
    input_ids_list, att_mask_list, labels_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    labels = torch.tensor(labels_list)
    
    if split=="train":
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, labels)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader

def get_data_dq(split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    images_list = []
    coords_list = []
    labels_list=[]
    cont = 0
    with open("jsons/tqa_dq.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_"+retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        images_q = []
        coords_q = []
        for count_i in range(4):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
            images_q.append(doc["image_path"])
            coord = [c[:4] for c in doc["coords"]]
            coords_q.append(coord)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images_list.append(images_q)
        coords_list.append(coords_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, images_list, coords_list, labels_list]

def process_data_tf(raw_data, batch_size, split):
    input_ids_list, att_mask_list, labels_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    labels = torch.tensor(labels_list)
    
    if split=="train":
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, labels)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader

def training_tf(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model=False):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        pbar = tqdm(train_dataloader)
        for batch in pbar:  
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            loss,logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/tf_roberta_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_tf(model, val_dataloader, device)
        
    print("")
    print("Training complete!")
        
def validation_tf(model, val_dataloader, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    for batch in tqdm(val_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res
    
def training_ndq(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model, dataset_name):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        pbar = tqdm(train_dataloader)
        for batch in pbar:  
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            loss,logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/tmc_"+dataset_name+"_roberta_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_ndq(model, val_dataloader, device)
        
    print("")
    print("Training complete!")
        
    
def validation_ndq(model, val_dataloader, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    for batch in tqdm(val_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res

def training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device, save_model):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        input_ids_list, att_mask_list, images_list, coords_list, labels_list = raw_data_train

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        dataset_ids = list(range(len(labels_list)))
        random.shuffle(dataset_ids)
        batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
        pbar = tqdm(batched_ids)
        for batch_ids in pbar:  
            model.train()
            b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
            b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
            b_input_images = [x for y,x in enumerate(images_list) if y in batch_ids]
            b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
            b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords, labels=b_labels)
            
            loss,logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/dmc_dq_roberta_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_dq(model, raw_data_val, batch_size, device)
        
    print("")
    print("Training complete!")
    
def validation_dq(model, raw_data_val, batch_size, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")
    
    input_ids_list, att_mask_list, images_list, coords_list, labels_list = raw_data_val

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    dataset_ids = list(range(len(labels_list)))
    batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
    pbar = tqdm(batched_ids)
    for batch_ids in pbar:
        # Unpack the inputs from our dataloader
        b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
        b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
        b_input_images = [x for y,x in enumerate(images_list) if y in batch_ids]
        b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
        b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords, labels=b_labels)

        loss, logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res
    
def generate_interagreement_chart(feats, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts:
            list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max),len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            res = sum(x == y for x, y in zip(i_solver, j_solver))/len(i_solver)
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split+'_interagreement.png')
    
def generate_complementarity_chart(feats, labels, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts:
            list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max),len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            points = 0
            totals = 0
            for e1,e2,lab in zip(i_solver,j_solver,labels):
                if e1!=lab:
                    if e2==lab:
                        points = points + 1
                    totals = totals + 1
            res = points/totals
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split+'_complementarity.png')

def get_upper_bound(feats, labels):
    points = 0
    for e1,e2,e3,lab in zip(feats[0],feats[1],feats[2],labels):
        if np.argmax(e1)==lab:
            points = points + 1
        else:
            if np.argmax(e2)==lab:
                points = points + 1
            else:
                if np.argmax(e3)==lab:
                    points = points + 1
    upper_bound = points/len(labels)
    return upper_bound
        
def ensembler(feats_train, feats_test, labels_train, labels_test):
    softmax = torch.nn.Softmax(dim=1)
    
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i],soft_ft[i]])
                if lab==i:
                    list_of_labels.append(1)
                else:
                    list_of_labels.append(0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels))
    
    list_of_elems = []
    list_of_labels = []
    for feats1, feats2, feats3, lab in zip(feats_train[0], feats_train[1], feats_train[2], labels_train):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        possible_answers = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output=output1+output2+output3
            list_of_elems.append(output)
            if lab==i:
                list_of_labels.append(1)
            else:
                list_of_labels.append(0)
    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels)

    points = 0
    for feats1, feats2, feats3, lab in zip(feats_test[0], feats_test[1], feats_test[2], labels_test):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        possible_answers = []
        outs = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output=output1+output2+output3
            outs.append(output)
        outs = [list(x) for x in outs]
        outs2 = final_model.predict_proba(outs)
        feats = [x[1] for x in outs2]
        outs3 = np.argmax(feats)
        if outs3==lab:
            points = points + 1
    return points/len(labels_test)

def superensembler(feats_train, feats_test, labels_train, labels_test):
    softmax = torch.nn.Softmax(dim=1)
    
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i],soft_ft[i]])
                if lab==i:
                    list_of_labels.append(1)
                else:
                    list_of_labels.append(0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels))
    
    list_of_elems = []
    list_of_labels = []
    for feats1, feats2, feats3, feats4, feats5, feats6, lab in zip(feats_train[0], feats_train[1], feats_train[2], feats_train[3], feats_train[4], feats_train[5], labels_train):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        soft4 = list(softmax(torch.tensor([feats4]))[0].detach().cpu().numpy())
        soft5 = list(softmax(torch.tensor([feats5]))[0].detach().cpu().numpy())
        soft6 = list(softmax(torch.tensor([feats6]))[0].detach().cpu().numpy())
        possible_answers = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output4 = solvers[3].predict_proba([[feats4[i], soft4[i]]])[0]
            output5 = solvers[4].predict_proba([[feats5[i], soft5[i]]])[0]
            output6 = solvers[5].predict_proba([[feats6[i], soft6[i]]])[0]
            output=output1+output2+output3+output4+output5+output6
            list_of_elems.append(output)
            if lab==i:
                list_of_labels.append(1)
            else:
                list_of_labels.append(0)
    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels)

    points = 0
    for feats1, feats2, feats3, feats4, feats5, feats6, lab in zip(feats_test[0], feats_test[1], feats_test[2], feats_test[3], feats_test[4], feats_test[5], labels_test):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        soft4 = list(softmax(torch.tensor([feats4]))[0].detach().cpu().numpy())
        soft5 = list(softmax(torch.tensor([feats5]))[0].detach().cpu().numpy())
        soft6 = list(softmax(torch.tensor([feats6]))[0].detach().cpu().numpy())
        possible_answers = []
        outs = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output4 = solvers[3].predict_proba([[feats4[i], soft4[i]]])[0]
            output5 = solvers[4].predict_proba([[feats5[i], soft5[i]]])[0]
            output6 = solvers[5].predict_proba([[feats6[i], soft6[i]]])[0]
            output=output1+output2+output3+output4+output5+output6
            outs.append(output)
        outs = [list(x) for x in outs]
        outs2 = final_model.predict_proba(outs)
        feats = [x[1] for x in outs2]
        outs3 = np.argmax(feats)
        if outs3==lab:
            points = points + 1
    return points/len(labels_test)