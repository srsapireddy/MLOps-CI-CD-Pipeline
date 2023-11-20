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
• Serving issues
• Built-in analysis
![image](https://github.com/srsapireddy/MLOps-CI-CD-Pipeline/assets/32967087/b1f2bc10-13da-457e-b416-005d0302bc3c)
• Monitoring schedule
  - Monitor your ML models by scheduling monitoring jobs through Amazon SageMaker Model Monitor.
  - Automatically kick off monitoring jobs to analyze model predictions during a given period.
  - Multiple schedules on a SageMaker endpoint.
• Reports and alerts
  - • Generated report by monitoring jobs can be saved in Amazon S3 for further analysis.
  - View model metrics via Amazon CloudWatch, consume notifications to trigger alarms or corrective actions, such as retraining the model or auditing data
  - Integrates with other visualization tools, including Tensorboard, Amazon QuickSight, and Tableau.

## Feature Store
































