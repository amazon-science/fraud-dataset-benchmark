# TO BE UPDATED BY USER
IAM_ROLE = "<IAM ROLE with acceess to S3 bucket containing the data and access to Amazon Fraud Detector>"
BUCKET = "<S3 bucket containing the data>"
KEY = "<Path of S3 file containing train from FDB data loader>"
MODEL_NAME = "<Model name that you want to give>"  # lower case alphanumeric only, only _ allowed as delimiter
MODEL_TYPE     = "ONLINE_FRAUD_INSIGHTS" # or TRANSACTION_FRAUD_INSIGHTS

import os
import time
import json
import boto3
import click
import string
import random
import logging
import pandas as pd


MODEL_DESC     = "Benchmarking model"
EVENT_DESC     = "Event for benchmarking model"
ENTITY_TYPE    = "user"  # this is provided in the dummy data. Will need to change if using different data
ENTITY_DESC    = "Entity for benchmarking model"

BATCH_PREDICTION_JOB = DETECTOR_NAME = EVENT_TYPE = MODEL_NAME  # Others are kept same as model name

# boto3 connections
client = boto3.client('frauddetector') 
s3 = boto3.client('s3')

@click.command()
@click.argument("config", type=click.Path(exists=True))
def afd_train_model_demo(config):
    
    #############################################
    #####               Setup               #####
    with open(config, "r") as f:
        config_file = json.load(f)
    
    
    EVENT_VARIABLES = [variable["variable_name"] for variable in config_file["variable_mappings"]]
    EVENT_LABELS = [v for k,v in config_file["label_mappings"].items()]
    EVENT_LABELS = [item for sublist in EVENT_LABELS for item in sublist]  # flattening list of lists

    # Variable mappings of demo data in this use case.  Important to teach this to customer
    click.echo(f'{pd.DataFrame(config_file["variable_mappings"])}')
    click.echo(f'{pd.DataFrame(config_file["label_mappings"])}')

    S3_DATA_PATH = "s3://" + os.path.join(BUCKET, KEY)
       
    #############################################
    ##### Create event variables and labels #####
    
    # -- create variable  --
    for variable in config_file["variable_mappings"]:
        
        DEFAULT_VALUE = '0.0' if variable["data_type"] == "FLOAT" else '<null>'
        
        try:
            resp = client.get_variables(name = variable["variable_name"])
            click.echo("{0} exists, data type: {1}".format(variable["variable_name"], resp['variables'][0]['dataType']))
        except:
            click.echo("Creating variable: {0}".format(variable["variable_name"]))
            resp = client.create_variable(
                    name         = variable["variable_name"],
                    dataType     = variable["data_type"],
                    dataSource   ='EVENT',
                    defaultValue = DEFAULT_VALUE, 
                    description  = variable["variable_name"],
                    variableType = variable["variable_type"])
    # Putting FRAUD
    for f in config_file["label_mappings"]["FRAUD"]:
        response = client.put_label(
            name = f,
            description = "FRAUD")
    # Putting LEGIT
    for f in config_file["label_mappings"]["LEGIT"]:
        response = client.put_label(
            name = f,
            description = "LEGIT")

    #############################################
    #####   Define Entity and Event Types   #####
    
    # -- create entity type --
    try:
        response = client.get_entity_types(name = ENTITY_TYPE)
        click.echo("-- entity type exists --")
        click.echo(response)
    except:
        response = client.put_entity_type(
            name        = ENTITY_TYPE,
            description = ENTITY_DESC
        )
        click.echo("-- create entity type --")
        click.echo(response)


    # -- create event type --
    try:
        response = client.get_event_types(name = EVENT_TYPE)
        click.echo("\n-- event type exists --")
        click.echo(response)
    except:
        response = client.put_event_type (
            name           = EVENT_TYPE,
            eventVariables = EVENT_VARIABLES,
            labels         = EVENT_LABELS,
            entityTypes    = [ENTITY_TYPE])
        click.echo("\n-- create event type --")
        click.echo(response)

    #############################################
    #####   Batch import training file for TFI  #####
    if MODEL_TYPE == "TRANSACTION_FRAUD_INSIGHTS":
        try:
            response = client.create_batch_import_job(
                jobId = BATCH_PREDICTION_JOB,
                inputPath = S3_DATA_PATH,
                outputPath = "s3://" + BUCKET,
                eventTypeName = EVENT_TYPE,
                iamRoleArn = IAM_ROLE
            )   
        except Exception:
            pass

        # -- wait until batch import is finished --
        print("--- waiting until batch import is finished ")
        stime = time.time()
        while True:
            response = client.get_batch_import_jobs(jobId=BATCH_PREDICTION_JOB)
            if 'IN_PROGRESS' in response['batchImports'][0]['status']:
                print(f"current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
                time.sleep(60)  # sleep for 1 minute 
            else:
                print("Batch Impoort status : " +  response['batchImports'][0]['status'])
                break

        etime = time.time()
        print(f"Elapsed time: {(etime - stime)/60:{3}.{3}} minutes \n"  )
        print(response)


    #############################################
    #####   Create and train your model     #####
    try:
        response = client.create_model(
           description   = MODEL_DESC,
           eventTypeName = EVENT_TYPE,
           modelId       = MODEL_NAME,
           modelType     = MODEL_TYPE)
        click.echo("-- initalize model --")
        click.echo(response)
    except Exception:
        pass
    
    # -- initalized the model, it's now ready to train --
    
    # -- first define training_data_schema for model to use --

    
    if MODEL_TYPE == "TRANSACTION_FRAUD_INSIGHTS": 
        training_data_schema = {
            'modelVariables' : EVENT_VARIABLES,
            'labelSchema'    : {
                'labelMapper' : config_file["label_mappings"],
                'unlabeledEventsTreatment': 'IGNORE'
            }
        }
        response = client.create_model_version(
            modelId             = MODEL_NAME,
            modelType           = MODEL_TYPE,
            trainingDataSource  = 'INGESTED_EVENTS',
            trainingDataSchema  = training_data_schema,
            ingestedEventsDetail={  # This needs to be changed
                  'ingestedEventsTimeWindow': {
                      'startTime': '2020-12-10T00:00:00Z', # '2021-08-28T00:00:00Z',
                      'endTime': '2022-06-07T00:00:00Z'  #'2022-05-10T00:00:00Z'
                  }
    }
        )
    else:
        training_data_schema = {
            'modelVariables' : EVENT_VARIABLES,
            'labelSchema'    : {
                'labelMapper' : config_file["label_mappings"]
            }
        }
        response = client.create_model_version(
            modelId             = MODEL_NAME,
            modelType           = MODEL_TYPE,
            trainingDataSource  = 'EXTERNAL_EVENTS',
            trainingDataSchema  = training_data_schema,
            externalEventsDetail = {
                'dataLocation'     : S3_DATA_PATH,
                'dataAccessRoleArn': IAM_ROLE
            }
        )
    model_version = response['modelVersionNumber']
    click.echo("-- model training --")
    click.echo(response)


if __name__=="__main__":
    afd_train_model_demo()
