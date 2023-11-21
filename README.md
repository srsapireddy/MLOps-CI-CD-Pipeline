# End-to-End-MLOps-CI-CD-Pipeline

## AWS SageMaker
Creating a notebook instance in AWS SageMaker </br></br>
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/5c06d7cb-4f3e-4506-88f3-18029f04fcb9)

### Upload the Notebook to Notebook Instance
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/78683c43-422e-47fc-aa1a-d4a6d152a88e)


### Change Domains in Admin Configuration
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/f87c68ed-90c4-491c-b121-967bdc07ce95)
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/a5f79bdf-9a08-4617-b8ce-645144627fe1)

### Inside the created domain: Launch the Studio
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/80b52c25-6ab6-46c4-a7c8-97534486ec1d)

* SageMaker Studio: This is an end-to-end MLOps Platform where we can build a level 4 architecture. This is where our MLOps Pipelines run. To run MLOps end-to-end pipeline. </br>
* Domain: It is a shared workspace we have created. The SageMaker domain is an integration of SageMaker Studio. </br>
* Notebook Instance: Notebook Instance is a standalone virtual machine (ml.m4.xlarge in our case) running anaconda on top of it where we have our Jupyter Notebooks. </br>

Our models are deployed to a SageMaker endpoint. And we need to invoke that endpoint. We want to avoid invoking the endpoint on the ASageMaker Studio itself. We created a different virtual machine for notebook instances to invoke ML Models. This ML model is deployed on the SageMaker End Point. Here, we have created a SageMaker Studio Domain to create a workspace.

### MLOps Benefits
* Reproducibility
* Deployment
* Monitoring: Monitoring in MLOps refers to observing, measuring, and analyzing the performance, health, and behavior of machine learning models and systems in a production environment. </br>

### Different Roles involved in MLOps
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/d1285e4d-3ffc-422d-8606-d427413e721a)

### DevOps vs MLOps
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/a55c66bd-9bdf-4781-b328-bb2aae97f4d9)

### Different Tools for MLOps
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/2767f1aa-6c91-4b27-bdd4-e5ff0da92fe0)

### Key Features of MLOps Tools
* Model training, tuning, and drift management
* Pipeline management
* Collaboration and communication capabilities

### AWS ML Stack
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/afaef614-28a9-4f26-9885-5365d8a4cfc9)

### AWS SageMaker: Integrated APIs
* Data scientists and developers can quickly and easily build and train machine learning models and then directly deploy them into a production-ready hosted environment
* Integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis
* Zero setups for data exploration
* Algorithms designed for huge datasets
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/6dca5ab7-6839-4824-8db0-dad05363297f)

### In Built Algorithms in SageMaker
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/e76c8429-aa3b-40f3-b07b-d522d5974811)
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/610ca717-133c-4b73-9fe8-b10fcadb6a6d)
#### ML Frameworks Supported
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/d1aa2e74-333d-449a-a944-f1e37f31969a)

### The following diagram shows how you train and deploy a model with Amazon SageMaker:
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/94389888-db0a-43c3-a160-022e0648a1d8)

The client data will be stored in the S3 bucket as our storage platform. </br>
The Amazon SageMaker is divided into three blocks:</br>
1. Model training: Our data scientists will write two kinds of scripts. </br>
   a. Helper Code: Define where the data comes from and how we are preprocessing the data. What kind of algorithm and hyperparameter tuning we are using are defined here.</br>
   b. Training Code: Nothing but the algorithm. AWS has all the training images in the ECR container registry (no need to pip install the packages). We need to pull the docker image to start doing the model training.
   Once the model training is done automatically, the ML code will be saved into the S3 bucket in the model.tar.gz file format.</br>
2. Deployment/ Hosting
   a. Helper Code: We mention to which machine we deploy our ML Model. How many devices are we deploying our model to? </br>
   b. Inference Code: This code is fetched from the ECR Container registry, which will be responsible for creating the endpoint. End Point is an API developers use to create client-side applications, which will be our REST API. </br>
3. EC2/ DCR Container Registry: Place where the dockerized images are stored. </br>
Amazon SageMaker can be integrated with AWS ground truth to collect the incoming data from the customers. We can reuse this data to retrain the model after 6 months to deploy another version of the ML model.

## AWS for MLOps
### Amazon SageMaker Pipelines
• First purpose-built CI/CD service for machine learning
• With SageMaker Pipelines, you can create, automate, and manage end-to-end ML workflows at scale.
• Orchestrating workflows across each step of the machine-learning process

### Amazon SageMaker Studio
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/2ef4164e-04fa-4afc-9bc4-638bfc3cc3fa)

Here, the domain is the shared space. Inside this domain, we can have a range of engineering teams: a data engineering team with 2 users and a data scientist team with 3 users. In this way, we can add multiple users and assign required permissions.

### Key Features: 
* Compose, manage, and reuse ML workflows
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/055a38db-7852-4f5f-9c80-ffb572e3642f)
* Choose the best models for deploying into production
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/8714f95a-9c12-4b9b-985f-96974f8d3370)
* Automatic tracking of models: AWS CloudWatch
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/51ca334f-b650-41c0-a444-f1f7e38ab06b)
* Bring CI/CD to machine learning
![image](https://github.com/srsapireddy/End-to-End-MLOps-CI-CD-Pipeline/assets/32967087/e3569a26-bb08-4911-ba7c-09cd7563651d)

### ML Model Monitoring
• Data drift and concept drift: Data collection and monitoring
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/31fd732d-fcf9-49ff-97bf-068cf06bc6bc)
• Serving issues  </br>
• Built-in analysis  </br>
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/b1f2bc10-13da-457e-b416-005d0302bc3c)
• Monitoring schedule
  - Monitor your ML models by scheduling monitoring jobs through Amazon SageMaker Model Monitor.
  - Automatically kick off monitoring jobs to analyze model predictions during a given period.
  - Multiple schedules on a SageMaker endpoint. 
• Reports and alerts
  - Generated reports by monitoring jobs can be saved in Amazon S3 for further analysis.
  - View model metrics via Amazon CloudWatch, consume notifications to trigger alarms or corrective actions, such as retraining the model or auditing data
  - Integrates with other visualization tools, including Tensorboard, Amazon QuickSight, and Tableau.
  - Use AWS EventBridge Service to trigger a timer to run the pipelines on SageMaker.
    
## Feature Store
Centralized repository or a platform for managing and serving machine learning features. </br>

### Key Benefits 
• Improved model accuracy and consistency   </br>
• Faster model development and deployment  </br>
• Better governance and compliance  </br>
• Increased collaboration and knowledge sharing  </br>
 
### Amazon SageMaker Feature Store
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/771886a4-a09f-4f10-9786-90ff8196719a)
#### Ingest data from many sources
✓ Ingest features using streaming data sources like Amazon Kinesis Data Firehose </br>
✓ Create features using data preparation tools such as Amazon SageMaker Data Wrangler & and store them directly in SageMaker Feature Store </br>
#### Search and discovery
✓ Tags and index features </br>
✓ Browsing the feature catalog </br>
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/af22fdae-c28f-4579-b06e-c524a7bb7347)

## AWS MLOps - Post Deployment Challenges
* Data Drift: Data drift is the change in data distribution over time.
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/7e334030-3dd6-4a7b-8b2d-1c8be7e9ddf9)
• We should probably retrain our model as we can see that Movie or TV streaming service is getting more and more popular with time. </br>
• Example: Iris flower dataset (Credits: Evidently AI)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/d74657e1-65c1-4ef1-943c-1e3395904f8f)
* Concept Drift: After some time, we can observe that the importance of different features changes with time. So, the model needs to be re-trained
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/26ff464c-153d-472b-96f0-50e808642471)
  - Loading time is getting more critical for churning with time.

## Software engineering challenges
• Environmental Changes: Some libraries used may run out of support. </br>
• Out of service Cloud </br>
• Compute resources (CPU/GPU/memory) </br>
• Security and Privacy </br>

### LAB
Launch a template for model building, training, and deployment </br>
#### SageMaker Studio
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/47debfb4-a2db-4be1-86a0-fced6787caea)
### Deploy a template in SageMaker for model building, training, and deploying: Root Directory of the project
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/665e1115-34d7-45b8-a63f-7d4230de49c2)

This will create two repositories for model build and model deployment. And integrate that with CI/CD. </br>
Once the pipeline is deployed using AWS DevOps tools from CI/CD. Here, we will approve our ML model for development and production environment. And we trigger our pipeline programmatically. </br>
* Triggering ML Pipelines:
  - SageMaker Python SDK
  - AWS EventBridge
  - Code Change

### Open CodeCommit: There are two repositories for model building and deployment.
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/a955e166-40ce-41c4-bbb0-bc9236dccbd9)

We need to pull or clone the repositories in CodeCommit to the SageMaker workspace. </br>
#### After Cloning:
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/3438f194-4193-4241-9b68-b2bf96bfbb05)

#### Executing the Pipeline
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/7164ef55-55c3-44cb-bad8-d42e24969aee)
#### Workflow Graph
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/6ad9aec4-7409-4a2f-add5-72c272fb1e48)


### Code Directory: Project Setup
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/8ad4818a-eb58-46a2-8411-6c315a1510a2)
#### Buildspec.yml file
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/958a6276-1180-430e-b7a8-4205c6298e27)

#### pipeline.py file
```
"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        step_args=step_args,
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    step_train = TrainingStep(
        name="TrainAbaloneModel",
        step_args=step_args,
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="RegisterAbaloneModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
```

#### preprocess.py file
```
"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)

    logger.debug("Defining transformers.")
    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
```

### evaluation.py file
```
"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.debug("Calculating mean squared error.")
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
```

### Process Tab
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/859f0790-7fd0-4acb-b197-9f34020b4000)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/893e8386-ebd5-4254-b78b-97a0551737d8)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/45cb1fcc-c84d-46b1-a358-83fc9f791251)

We can click the Tabs after executing the pipeline to check the logs </br>

### Training Tab
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/ab38f9c2-a7c9-4633-8f48-5ab9a44f60a4)

### Model Registry Tab
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/c8ca0772-22a1-41ab-8190-cfbe10d0a089)

## Deploying Model to Development Environment
Go to Model Groups </br>
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/242d3d22-7012-45d0-a400-0230433e2d02)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/aeb7c38d-3594-400a-ba4d-8cd692bd6ab5)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/df2a0059-a631-4cfb-8456-193575eb67cd)
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/4741557f-434a-4c6a-a0dd-a2a3ef3862e2)

Here, we can see that an endpoint is created. That is how our model is getting deployed.
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/27883ddc-c8b7-4918-a32b-5f2fe65a9515)
Deployed Model
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/05b3399f-badf-4c7c-8460-c04bd24c9db9)

You can use the code I've included below to invoke the deployed model.
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/ffcc981c-d367-4e1c-8f85-7ef538ca4a5c)
Change the endpoint name to "mlops-nov-masterclass-staging"
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/fd103dd1-f41f-4feb-b868-aa15a47e5f75)


## Model Deploy Repository
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/8c90b285-9cfd-46de-89ba-59f6c911a362)
Here, we take the model from the model registry and deploy it to the production </br>
### buildspec.yml file:
```
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      # Upgrade AWS CLI to the latest version
      - pip install --upgrade --force-reinstall "botocore>1.21.30" "boto3>1.18.30" "awscli>1.20.30"

  build:
    commands:
      # Export the staging and production configuration files
      - python build.py --model-execution-role "$MODEL_EXECUTION_ROLE_ARN" --model-package-group-name "$SOURCE_MODEL_PACKAGE_GROUP_NAME" --sagemaker-project-id "$SAGEMAKER_PROJECT_ID" --sagemaker-project-name "$SAGEMAKER_PROJECT_NAME" --s3-bucket "$ARTIFACT_BUCKET" --export-staging-config $EXPORT_TEMPLATE_STAGING_CONFIG --export-prod-config $EXPORT_TEMPLATE_PROD_CONFIG

      # Package the infrastucture as code defined in endpoint-config-template.yml by using AWS CloudFormation.
      # Note that the Environment Variables like ARTIFACT_BUCKET, SAGEMAKER_PROJECT_NAME etc,. used below are expected to be setup by the
      # CodeBuild resrouce in the infra pipeline (in the ServiceCatalog product)
      - aws cloudformation package --template endpoint-config-template.yml --s3-bucket $ARTIFACT_BUCKET --output-template $EXPORT_TEMPLATE_NAME

      # Print the files to verify contents
      - cat $EXPORT_TEMPLATE_STAGING_CONFIG
      - cat $EXPORT_TEMPLATE_PROD_CONFIG

artifacts:
  files:
    - $EXPORT_TEMPLATE_NAME
    - $EXPORT_TEMPLATE_STAGING_CONFIG
    - $EXPORT_TEMPLATE_PROD_CONFIG
```



## Use Case: Predicting the age of abalone
Train the ML Model to predict the age of an abalone. </br>
Reference: https://anson.ucdavis.edu/~haochen/abalone_description.pdf



































