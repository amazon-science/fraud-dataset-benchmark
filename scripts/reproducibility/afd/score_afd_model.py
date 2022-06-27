# TO BE UPDATED BY USER
IAM_ROLE = "<IAM ROLE with acceess to S3 bucket containing the data and access to Amazon Fraud Detector>"
BUCKET = "<S3 BUCKET>"
TEST_PATH = "<Path of S3 file containing test from FDB data loader>"
TEST_LABELS_PATH = "<Path of S3 file containing test_labels from FDB data loader>"
MODEL_NAME = "<Name of trained model to be used for scoring on the test data>"  # lower case alphanumeric only, only _ allowed as delimiter
MODEL_TYPE     = "ONLINE_FRAUD_INSIGHTS" # or TRANSACTION_FRAUD_INSIGHTS

import os
import ast
import time
import json
import boto3
import click
import string
import random
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# boto3 connections
client = boto3.client('frauddetector') 
s3 = boto3.client('s3')

BATCH_PREDICTION_JOB = DETECTOR_NAME = EVENT_TYPE = MODEL_NAME
model_version = '1.0'
DETECTOR_DESC = "Benchmarking detector"


def create_outcomes(outcomes):
    """ 
    Create Fraud Detector Outcomes 
    """   
    for outcome in outcomes:
        print("creating outcome variable: {0} ".format(outcome))
        response = client.put_outcome(name = outcome, description = outcome)


def create_rules(score_cuts, outcomes):
    """
    Creating rules 
    
    Arguments:
        score_cuts  - list of score cuts to create rules
        outcomes    - list of outcomes associated with the rules
    
    Returns:
        a rule list to used when create detector
    """
    
    if len(score_cuts)+1 != len(outcomes):
        logging.error('Your socre cuts and outcomes are not matched.')
    
    rule_list = []
    for i in range(len(outcomes)):
        # rule expression
        if i < (len(outcomes)-1):
            rule = "${0}_insightscore > {1}".format(MODEL_NAME,score_cuts[i])
        else:
            rule = "${0}_insightscore <= {1}".format(MODEL_NAME,score_cuts[i-1])
    
        # append to rule_list (used when create detector)
        rule_id = "rules{0}_{1}".format(i, MODEL_NAME[:9])
        
        rule_list.append({
            "ruleId": rule_id, 
            "ruleVersion" : '1',
            "detectorId"  : DETECTOR_NAME
        })
        
        # create rules
        print("creating rule: {0}: IF {1} THEN {2}".format(rule_id, rule, outcomes[i]))
        try:
            response = client.create_rule(
                ruleId = rule_id,
                detectorId = DETECTOR_NAME,
                expression = rule,
                language = 'DETECTORPL',
                outcomes = [outcomes[i]]
                )
        except:
            print("this rule already exists in this detector")
            
    return rule_list


def ast_with_nan(x):
    try:
        return ast.literal_eval(x)
    except:
        return np.nan


def afd_train_model_demo():

    # -- activate the model version --
    try:
        response = client.update_model_version_status (
            modelId            = MODEL_NAME,
            modelType          = MODEL_TYPE,
            modelVersionNumber = model_version,
            status             = 'ACTIVE'
        )
        print("-- activating model --")
        print(response)
    except Exception:
        print("First train the model")
    
    # -- wait until model is active --
    print("--- waiting until model status is active ")
    stime = time.time()
    while True:
        response = client.get_model_version(modelId=MODEL_NAME, modelType = MODEL_TYPE, modelVersionNumber = model_version)
        if response['status'] != 'ACTIVE':
            print(response['status'])
            print(f"current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
            time.sleep(60)  # sleep for 1 minute 
        if response['status'] == 'ACTIVE':
            print("Model status : " +  response['status'])
            break
            
    etime = time.time()
    print("Elapsed time : %s" % (etime - stime) + " seconds \n"  )
    print(response)

    # -- put detector, initalizes your detector -- 
    response = client.put_detector(
        detectorId    = DETECTOR_NAME, 
        description   = DETECTOR_DESC,
        eventTypeName = EVENT_TYPE )

    # -- decide what threshold and corresponding outcome you want to add -- 
    # here, we create three simple rules by cutting the score at [950,750], and create three outcome ['fraud', 'investigate', 'approve'] 
    # it will create 3 rules:
    #    score > 950: fraud
    #    score <= 750: approve

    score_cuts = [750]                          # recommended to fine tune this based on your business use case
    outcomes = ['fraud', 'approve']  # recommended to define this based on your business use case

    # -- create outcomes -- 
    print(" -- create outcomes --")
    create_outcomes(outcomes)

    # -- create rules --
    print(" -- create rules --")
    rule_list = create_rules(score_cuts, outcomes)

    # -- create detector version --
    client.create_detector_version(
        detectorId    = DETECTOR_NAME,
        rules         = rule_list,
        modelVersions = [{"modelId": MODEL_NAME, 
                        "modelType": MODEL_TYPE,
                        "modelVersionNumber": model_version}],
        # there are 2 options for ruleExecutionMode:
        #   'ALL_MATCHED'    - return all matched rules' outcome
        #   'FIRST_MATCHED'  - return first matched rule's outcome
        ruleExecutionMode = 'FIRST_MATCHED'
    )

    print("\n -- detector created -- ")
    print(response) 

    response = client.update_detector_version_status(
        detectorId        = DETECTOR_NAME,
        detectorVersionId = '1',
        status            = 'ACTIVE'
    )
    print("\n -- detector activated -- ")
    print(response)

    # -- wait until detector is active --
    print("\n --- waiting until detector status is active ")
    stime = time.time()
    while True:
        response = client.describe_detector(
            detectorId        = DETECTOR_NAME,
        )
        if response['detectorVersionSummaries'][0]['status'] != 'ACTIVE':
            print(response['detectorVersionSummaries'][0]['status'])
            print(f"current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
            time.sleep(60)
        if response['detectorVersionSummaries'][0]['status'] == 'ACTIVE':
            break
    etime = time.time()
    print("Elapsed time : %s" % (etime - stime) + " seconds \n"  )
    print(response)

    # -- create detector evaluation --
    try:
        client.create_batch_prediction_job (
        jobId = BATCH_PREDICTION_JOB,
        inputPath = os.path.join('s3://', BUCKET, TEST_PATH),
        outputPath =os.path.join('s3://', BUCKET),
        eventTypeName = EVENT_TYPE,
        detectorName = DETECTOR_NAME,
        detectorVersion = '1',
        iamRoleArn = IAM_ROLE)
    except Exception as e:
        print(e)
        print("batch prediction job already exists")

    # -- wait until batch prediction job is completed --
    print("\n --- waiting until batch prediction job is completed ")
    stime = time.time()
    while True:
        response = client.get_batch_prediction_jobs(jobId=BATCH_PREDICTION_JOB)
        response = response['batchPredictions'][0]
        if (response['status'] != 'COMPLETE') and (response['status'] != 'FAILED'):
            print(f"current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
            time.sleep(60)
        if response['status'] == 'COMPLETE':
            break
    etime = time.time()
    print("Elapsed time : %s" % (etime - stime) + " seconds \n"  )
    print(response)

    # -- get batch prediction job result --
    contents = s3.list_objects_v2(Bucket=BUCKET, Prefix=os.path.join(TEST_PATH))['Contents']
    print(contents)
    S3_SCORE_PATH = sorted([c['Key'] for c in contents if c['Key'].endswith('output.csv')])[-1]
    print(S3_SCORE_PATH)

    # -- get test performance --
    # Predictions
    print(os.path.join('s3://', BUCKET, S3_SCORE_PATH))
    predictions = pd.read_csv(os.path.join('s3://', BUCKET, S3_SCORE_PATH))
    predictions = predictions.copy()[~predictions.MODEL_SCORES.isna()]

    predictions['scores'] = predictions['MODEL_SCORES'].\
    apply(lambda x: ast_with_nan(x)).\
    apply(lambda x: x.get(MODEL_NAME))

    # Labels
    labels = pd.read_csv(os.path.join('s3://', BUCKET, TEST_LABELS_PATH))
#     labels['EVENT_LABEL'] = labels['EVENT_LABEL'].map({'benign': 0, 'malignant': 1})
    predictions = predictions.merge(labels, on='EVENT_ID', how='left')
    print('Test size: ', predictions.shape)

    fpr, tpr, threshold = roc_curve(predictions['EVENT_LABEL'], predictions['scores'])
    test_auc = auc(fpr,tpr)
    print('AUC: ', test_auc)

    test_metrics = {}
    test_metrics['auc'] = test_auc
    test_metrics['fpr'] = list(fpr)
    test_metrics['tpr'] = list(tpr)
    test_metrics['threshold'] = list(threshold)

    # -- put test metrics in s3 --
    s3.put_object(
        Body=json.dumps(test_metrics), 
        Bucket=BUCKET, 
        Key='test_metrics.json') 

    print("\n -- test metrics saved -- ")

if __name__ == "__main__":
    afd_train_model_demo()

        

    

    

    