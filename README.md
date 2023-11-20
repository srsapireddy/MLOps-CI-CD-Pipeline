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





































