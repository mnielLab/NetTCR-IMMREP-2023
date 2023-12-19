# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd 
import subprocess
import itertools
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import time, random
import argparse

#Silence Tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

seed=15

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile", help="Specifies the input file with peptide and all six CDR sequences")
parser.add_argument("-g", "--github_dir", help="Directory where the github repository has been downloaded to")
parser.add_argument("-m", "--model_path", help="Path to the txt document containing the list of models to include in the ensemble")
parser.add_argument("-tb", "--tcrbase_file", default = None, help="Predictions from TCRbase")
parser.add_argument("-o", "--outfile", help="Specifies the output file")
parser.add_argument("-a", "--alpha", default = 10, help="Determines how much the final predictions takes similarity to known binders into account via TCRbase.\nThe final prediction score is given by pred = CNN_pred*TCRbase_pred^alpha. An alpha of 0 disables TCRbase scaling")
args = parser.parse_args()

infile = str(args.infile)
alpha = int(args.alpha)
github_dir = str(args.github_dir)
model_path = str(args.model_path)
tcrbase_file = str(args.tcrbase_file)
outfile = str(args.outfile)

sys.path.append(os.path.normpath(os.path.join(github_dir, "src")))
sys.path.append(os.path.normpath(os.path.join(github_dir, "models")))

import keras_utils


### Model parameters ###
train_parts = {0, 1, 2, 3, 4} #Partitions
encoding = keras_utils.blosum50_20aa_masking #Encoding for amino acid sequences

#Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

# Set random seed
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
full_data = pd.read_csv(infile)


def my_numpy_function(y_true, y_pred):
    """Implementation of AUC 0.1 metric for Tensorflow"""
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Case when only one class exists
        auc = np.array([float(0)])
    return auc
    
def auc_01(y_true, y_pred):
    """Converts function to optimised tensorflow numpy function"""
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01
     
def make_tf_ds(df, encoding = keras_utils.blosum50_20aa_masking):
    """Encodes amino acid sequences using a BLOSUM50 matrix with a normalization factor of 5.
    Sequences are right-padded with [-1x20] for each AA missing, compared to the maximum embedding 
    length for that given feature
    
    Additionally, the input is prepared for predictions, by loading the data into a list of numpy arrays"""
    
    encoded_pep = keras_utils.enc_list_bl_max_len(df.Peptide, encoding, pep_max)/5
    encoded_a1 = keras_utils.enc_list_bl_max_len(df.CDR1a, encoding, a1_max)/5
    encoded_a2 = keras_utils.enc_list_bl_max_len(df.CDR2a, encoding, a2_max)/5
    encoded_a3 = keras_utils.enc_list_bl_max_len(df.CDR3a, encoding, a3_max)/5
    encoded_b1 = keras_utils.enc_list_bl_max_len(df.CDR1b, encoding, b1_max)/5
    encoded_b2 = keras_utils.enc_list_bl_max_len(df.CDR2b, encoding, b2_max)/5
    encoded_b3 = keras_utils.enc_list_bl_max_len(df.CDR3b, encoding, b3_max)/5
    tf_ds = [np.float32(encoded_pep),
             np.float32(encoded_a1), np.float32(encoded_a2), np.float32(encoded_a3), 
             np.float32(encoded_b1), np.float32(encoded_b2), np.float32(encoded_b3)]


    return tf_ds
    
### Get peptides in input file ###
pep_list = list(full_data.Peptide.value_counts(ascending=False).index)

#Prepare output DataFrame (test predictions)
full_pred_df = pd.DataFrame()

#Necessary to load the model with the custom metric
dependencies = {
    'auc_01': auc_01
}

if tcrbase_file is not None:
    tcrbase_df = pd.read_csv(tcrbase_file)
    
model_list = []
with open(model_path, "r") as infile: 
    model_list = infile.readlines()
model_list = [os.path.normpath(github_dir+x.strip()) for x in model_list]

print("Making predictions for ", outfile)
print("Using the following files:")
print("\n".join(model_list))
#Predictions
for pep in pep_list:
    time_start = time.time()
    pred_df = full_data[full_data.Peptide == pep].copy(deep = True)
    test_tensor = make_tf_ds(pred_df, encoding = encoding)
    
    tcrbase_pep_df = tcrbase_df[tcrbase_df.Peptide == pep].copy(deep = True)
    
    #Flag for scaling with TCRbase
    scale_prediction = False
    
    #Used for announcing that a model does not exist for the given peptide
    print_flag = 0
    
    print("Making predictions for {}".format(pep), end = "")
    if alpha != 0:
        if tcrbase_pep_df.TCRbase_flag.unique()[0] == 0:
            scale_prediction = False
        else:
            scale_prediction = True
    
    avg_prediction = 0
    n_models = 0
    for model in model_list:
        n_models += 20
        for t in train_parts:
            x_test = test_tensor[0:7]
            
            for v in train_parts:
                if v!=t:      
                    
                    # Load the TFLite model and allocate tensors.
                    try:
                        interpreter = tf.lite.Interpreter(model_path = os.path.normpath(model+'/pre_trained/'+pep+'/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".tflite"))
                    except ValueError as error:
                        #print(error)
                        print_flag += 1
                        # Load pan-specific TFLite model and allocate tensors.
                        interpreter = tf.lite.Interpreter(model_path = os.path.normpath(model+'/cdr123_pan/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".tflite"))
                        if print_flag == 1:
                            print(". WARNING: A model for {} does not exist. Using pan-specific model instead ".format(pep), end = "")
                    
                    # Get input and output tensors for the model.
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    #Fix Output dimensions
                    output_shape = output_details[0]['shape']
                    interpreter.resize_tensor_input(output_details[0]["index"], [x_test[0].shape[0], output_details[0]["shape"][1]])
                    
                    #Fix Input dimensions
                    for i in range(len(input_details)):
                        interpreter.resize_tensor_input(input_details[i]["index"], [x_test[0].shape[0], input_details[i]["shape"][1], input_details[i]["shape"][2]])
                    
                    #Prepare tensors
                    interpreter.allocate_tensors()
                    
                    data_dict = {"pep": x_test[0],
                                 "a1": x_test[1],
                                 "a2": x_test[2],
                                 "a3": x_test[3],
                                 "b1": x_test[4],
                                 "b2": x_test[5],
                                 "b3": x_test[6]}
                    
                    #Assign input data
                    for i in range(len(input_details)):   
                        #Set input data for a given feature based on the name of the input in "input_details"
                        interpreter.set_tensor(input_details[i]['index'], data_dict[input_details[i]["name"].split(":")[0].split("_")[-1]])
                    
                    #Prepare the model for predictions
                    interpreter.invoke()
    
                    #Predict on input tensor
                    avg_prediction += interpreter.get_tensor(output_details[0]['index'])
        
                    #Clears the session for the next model
                    tf.keras.backend.clear_session()
    
    #Averaging the predictions between all models
    avg_prediction = avg_prediction/n_models
    
    #Flatten list of predictions
    avg_prediction = list(itertools.chain(*avg_prediction))
    
    #Run TCRbase if alpha is not set to 0, and a positive database for the peptide exists
    if scale_prediction is True and tcrbase_file is not None:
        pred_df['Prediction'] = avg_prediction * tcrbase_pep_df["Prediction"] ** alpha  
    else:
        pred_df['Prediction'] = avg_prediction
    
    full_pred_df = pd.concat([full_pred_df, pred_df])
    print("- Predictions took "+str(round(time.time()-time_start, 3))+" seconds\n")
    
#Save predictions in the same order as the input
full_pred_df.sort_values("ID", ascending = True, inplace = True)
full_pred_df.to_csv(outfile, index=False, sep = ',')

#Print prediction to stdout
#print("\n \nBelow is a table represention of binding predictions between T-Cell receptors and peptides. \n \n")
#print(full_pred_df)