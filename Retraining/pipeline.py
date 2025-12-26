"""Training-only pipeline of Hubble Project.
"""
import os

import boto3
import logging
import sagemaker
import sagemaker.session
# from datetime import date  # needed to get today's date

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

from sagemaker.network import NetworkConfig   # needed to run data extraction step

from botocore.exceptions import ClientError


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


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

def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """ Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    #Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None

def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    project_id="SageMakerProjectId",
    processing_instance_type="ml.m5.4xlarge",
    training_instance_type="ml.m5.xlarge",
    inference_instance_type="ml.m5.xlarge",
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
#     input_data = ParameterString(
#         name="InputDataUrl",
# #         default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
#         default_value="s3://sagemaker-migration-hubble/data"
#     )
    processing_image_name = "sagemaker-{0}-processing-imagebuild".format(project_id)
    training_image_name = "sagemaker-{0}-training-imagebuild".format(project_id)
    inference_image_name = "sagemaker-{0}-inference-imagebuild".format(project_id)
    
    # today_date=date.today()  # needed to get today's date
    
    
    ################################################################################
    ################################################################################
    # data extraction step for training data
    ################################################################################
    ################################################################################
    try:
        processing_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=processing_image_name)['ContainerImage']
    except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
        processing_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.0-1",
            py_version="py3",
            instance_type=processing_instance_type,
        )
    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/hubble-data-extraction",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role='arn:aws:iam::318685047967:role/Role_Sagemaker',
        network_config=NetworkConfig(security_group_ids=['sg-0a9932248ac0ea55a', 'sg-0730ef060ad6ecf45'],
                                    subnets=['subnet-00e6236c99b7bce81','subnet-0287d47b678907dfd', 'subnet-0f422e73bc3febb31'])
        
    )
    step_args = script_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ],
        code=os.path.join(BASE_DIR, "etl_train.py"),
    )
    step_data_extraction_train = ProcessingStep(
        name="Data-Extraction-Train",
        step_args=step_args,
    )
    
    ################################################################################
    # data extraction step for testing data
    step_args = script_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "etl_test.py"),
    )
    step_data_extraction_test = ProcessingStep(
        name="Data-Extraction-Test",
        step_args=step_args,
    )
    
    ################################################################################
    # data extraction step for inference data
    """
    step_args = script_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(output_name="inference", source="/opt/ml/processing/inference"),
        ],
        code=os.path.join(BASE_DIR, "etl_inference.py"),
    )
    step_data_extraction_inference = ProcessingStep(
        name="Data-Extraction-Inference",
        step_args=step_args,
    )
    """
    
    ################################################################################
    ################################################################################
    # processing step for training data
    ################################################################################
    ################################################################################

    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/hubble-preprocess",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_data_extraction_train.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            # ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                source="/opt/ml/processing/train",
                output_name="train",
                destination="s3://sagemaker-migration-hubble/pipeline-active/data/",
                s3_upload_mode='EndOfJob',
                ),
            # save an archive
            ProcessingOutput(
                source="/opt/ml/processing/train_archive",
                output_name="train_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/data/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "preprocess-train.py"),
    )
    step_process_train = ProcessingStep(
        name="Preprocess-Training-Data",
        step_args=step_args,
    )
    
    ################################################################################
    # processing step for validation data

    step_args = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_data_extraction_test.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            # ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(
                source="/opt/ml/processing/test",
                output_name="test",
                destination="s3://sagemaker-migration-hubble/pipeline-active/data/",
                s3_upload_mode='EndOfJob',
                ),
            # save an archive
            ProcessingOutput(
                source="/opt/ml/processing/test_archive",
                output_name="test_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/data/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "preprocess-test.py"),
    )
    step_process_test = ProcessingStep(
        name="Preprocess-Testing-Data",
        step_args=step_args,
    )
    
    ################################################################################
    # processing step for inference data
    """
    step_args = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_data_extraction_inference.properties.ProcessingOutputConfig.Outputs[
                    "inference"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="inference", source="/opt/ml/processing/inference"),
        ],
        code=os.path.join(BASE_DIR, "preprocess-inference.py"),
    )
    step_process_inference = ProcessingStep(
        name="Preprocess-Inference-Data",
        step_args=step_args,
    )
    """
    
    ################################################################################
    ################################################################################
    ################################################################################
    # Feature selection, Model selection and hyperparameter tuning step for smallfleet
    # filename_hyperparameter_smallfleet="hyperparameter_tuning_smallfleet_"+str(today_date)
    filename_hyperparameter_smallfleet="hyperparameter_tuning_smallfleet"
    step_args_small = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_test.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            # ProcessingOutput(output_name="feature_selection_smallfleet", source="/opt/ml/processing/feature_selection_smallfleet"),
            # ProcessingOutput(output_name="model_selection_smallfleet", source="/opt/ml/processing/model_selection_smallfleet"),
            # ProcessingOutput(output_name="feature_dict_smallfleet", source="/opt/ml/processing/feature_dict_smallfleet"),
            # ProcessingOutput(output_name="hyperparameter_tuning_smallfleet", source="/opt/ml/processing/hyperparameter_tuning_smallfleet"),
            
            # origin
            ProcessingOutput(
                source="/opt/ml/processing/feature_selection_smallfleet",
                output_name="feature_selection_smallfleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/model_selection_smallfleet",
                output_name="model_selection_smallfleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_smallfleet",
                output_name="feature_dict_smallfleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_smallfleet",
                output_name="hyperparameter_tuning_smallfleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            
            # archive
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_smallfleet_archive",
                output_name="feature_dict_smallfleet_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_smallfleet_archive",
                output_name="hyperparameter_tuning_smallfleet_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "refinement_smallfleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_smallfleet = ProcessingStep(
        name="Model_dev_smallfleet",
        step_args=step_args_small,
    )
   

    ################################################################################
    # Feature selection, Model selection and hyperparameter tuning step for largefleet
    # filename_hyperparameter_largefleet="hyperparameter_tuning_largefleet_"+str(today_date)
    filename_hyperparameter_largefleet="hyperparameter_tuning_largefleet"
    step_args_large = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_test.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            # ProcessingOutput(output_name="feature_selection_largefleet", source="/opt/ml/processing/feature_selection_largefleet"),
            # ProcessingOutput(output_name="model_selection_largefleet", source="/opt/ml/processing/model_selection_largefleet"),
            # ProcessingOutput(output_name="feature_dict_largefleet", source="/opt/ml/processing/feature_dict_largefleet"),
            # ProcessingOutput(output_name="hyperparameter_tuning_largefleet", source="/opt/ml/processing/hyperparameter_tuning_largefleet"),
            
            # origin
            ProcessingOutput(
                source="/opt/ml/processing/feature_selection_largefleet",
                output_name="feature_selection_largefleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/model_selection_largefleet",
                output_name="model_selection_largefleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_largefleet",
                output_name="feature_dict_largefleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_largefleet",
                output_name="hyperparameter_tuning_largefleet",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            
            # archive
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_largefleet_archive",
                output_name="feature_dict_largefleet_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_largefleet_archive",
                output_name="hyperparameter_tuning_largefleet_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "refinement_largefleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_largefleet = ProcessingStep(
        name="Model_dev_largefleet",
        step_args=step_args_large,
    )
    
    #Feature selection, Model selection and hyperparameter tuning step for expresscheck
    # filename_hyperparameter_expresscheck="hyperparameter_tuning_expresscheck_"+str(today_date)
    filename_hyperparameter_expresscheck="hyperparameter_tuning_expresscheck"
    step_args_express = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_test.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
    
        ],
        outputs=[
            # ProcessingOutput(output_name="feature_selection_expresscheck", source="/opt/ml/processing/feature_selection_expresscheck"),
            # ProcessingOutput(output_name="model_selection_expresscheck", source="/opt/ml/processing/model_selection_expresscheck"),
            # ProcessingOutput(output_name="feature_dict_expresscheck", source="/opt/ml/processing/feature_dict_expresscheck"),
            # ProcessingOutput(output_name="hyperparameter_tuning_expresscheck", source="/opt/ml/processing/hyperparameter_tuning_expresscheck"),
            
            # origin
            ProcessingOutput(
                source="/opt/ml/processing/feature_selection_expresscheck",
                output_name="feature_selection_expresscheck",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/model_selection_expresscheck",
                output_name="model_selection_expresscheck",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_expresscheck",
                output_name="feature_dict_expresscheck",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_expresscheck",
                output_name="hyperparameter_tuning_expresscheck",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            
            # archive
            ProcessingOutput(
                source="/opt/ml/processing/feature_dict_expresscheck_archive",
                output_name="feature_dict_expresscheck_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/hyperparameter_tuning_expresscheck_archive",
                output_name="hyperparameter_tuning_expresscheck_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "refinement_expresscheck.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_expresscheck = ProcessingStep(
        name="Model_dev_expresscheck",
        step_args=step_args_express,
    )

###################################################################################    
###################################################################################    
###################################################################################    
    #Training step for express check
###################################################################################    
###################################################################################    
    step_args_express_train = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_expresscheck.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_expresscheck].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_expresscheck",
            ),
            ProcessingInput(
                source=step_process_expresscheck.properties.ProcessingOutputConfig.Outputs["feature_dict_expresscheck"].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_expresscheck",
            ),
    
        ],
        outputs=[
            # ProcessingOutput(output_name="expresscheck_model_copy", source="/opt/ml/processing/expresscheck_model_copy"),
            ProcessingOutput(
                source="/opt/ml/processing/expresscheck_model",
                output_name="expresscheck_model",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/expresscheck_model_archive",
                output_name="expresscheck_model_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/expresscheck/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "train_expresscheck.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_expresscheck_train = ProcessingStep(
        name="Train_expresscheck",
        step_args=step_args_express_train,
    )
    
        
    #Training step for largefleet
    step_args_largefleet_train = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_largefleet.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_largefleet].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_largefleet",
            ),
            ProcessingInput(
                source=step_process_largefleet.properties.ProcessingOutputConfig.Outputs["feature_dict_largefleet"].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_largefleet",
            ),
    
        ],
        outputs=[
            # ProcessingOutput(output_name="largefleet_model_copy", source="/opt/ml/processing/largefleet_model_copy"),
            ProcessingOutput(
                source="/opt/ml/processing/largefleet_model",
                output_name="largefleet_model",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/largefleet_model_archive",
                output_name="largefleet_model_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/largefleet/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "train_largefleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_largefleet_train = ProcessingStep(
        name="Train_largefleet",
        step_args=step_args_largefleet_train,
    )
    
    #Training step for smallfleet
    step_args_smallfleet_train = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_train.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process_smallfleet.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_smallfleet].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_smallfleet",
            ),
            ProcessingInput(
                source=step_process_smallfleet.properties.ProcessingOutputConfig.Outputs["feature_dict_smallfleet"].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_smallfleet",
            ),
    
        ],
        outputs=[
            # ProcessingOutput(output_name="smallfleet_model_copy", source="/opt/ml/processing/smallfleet_model_copy"),
            ProcessingOutput(
                source="/opt/ml/processing/smallfleet_model",
                output_name="smallfleet_model",
                destination="s3://sagemaker-migration-hubble/pipeline-active/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
            ProcessingOutput(
                source="/opt/ml/processing/smallfleet_model_archive",
                output_name="smallfleet_model_archive",
                destination="s3://sagemaker-migration-hubble/pipeline-archive/model/smallfleet/",
                s3_upload_mode='EndOfJob',
                ),
        ],
        code=os.path.join(BASE_DIR, "train_smallfleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_smallfleet_train = ProcessingStep(
        name="Train_smallfleet",
        step_args=step_args_smallfleet_train,
    )
    
    
    """
    #################################################################################
    #Inference step for expresscheck
    step_args_express_inference = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_inference.properties.ProcessingOutputConfig.Outputs[
                    "inference"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/inference",
            ),
            ProcessingInput(
                source= step_process_expresscheck.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_expresscheck].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_expresscheck",
            ),
            ProcessingInput(
                source=step_process_expresscheck.properties.ProcessingOutputConfig.Outputs[
                    "feature_dict_expresscheck"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_expresscheck",
            ),
            ProcessingInput(
                source=step_process_expresscheck_train.properties.ProcessingOutputConfig.Outputs[
                    "expresscheck_model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/expresscheck_model",
            ),
    
        ],
        # outputs=[
        #     ProcessingOutput(output_name="express_df", source="/opt/ml/processing/express_df"),
        # ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/express_df",
                output_name="express_df",
                destination="s3://sagemaker-migration-hubble/model-output/express_df/",
                s3_upload_mode='EndOfJob',
                )
        ],
        code=os.path.join(BASE_DIR, "inference_expresscheck.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_expresscheck_inference = ProcessingStep(
        name="Inference_expresscheck",
        step_args=step_args_express_inference,
    )
    
    #################################################################################
    #Inference step for largefleet
    
    step_args_large_inference = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_inference.properties.ProcessingOutputConfig.Outputs[
                    "inference"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/inference",
            ),
            ProcessingInput(
                source= step_process_largefleet.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_largefleet].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_largefleet",
            ),
            ProcessingInput(
                source=step_process_largefleet.properties.ProcessingOutputConfig.Outputs[
                    "feature_dict_largefleet"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_largefleet",
            ),
            ProcessingInput(
                source=step_process_largefleet_train.properties.ProcessingOutputConfig.Outputs[
                    "largefleet_model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/largefleet_model",
            ),
    
        ],
        # outputs=[
        #     ProcessingOutput(output_name="large_df", source="/opt/ml/processing/large_df"),
        # ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/large_df",
                output_name="large_df",
                destination="s3://sagemaker-migration-hubble/model-output/large_df/",
                s3_upload_mode='EndOfJob',
                )
        ],
        code=os.path.join(BASE_DIR, "inference_largefleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_largefleet_inference = ProcessingStep(
        name="Inference_largefleet",
        step_args=step_args_large_inference,
    )

    #################################################################################
    #Inference step for smallfleet
    
    step_args_small_inference = script_processor.run(
        inputs=[
            ProcessingInput(
                source=step_process_inference.properties.ProcessingOutputConfig.Outputs[
                    "inference"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/inference",
            ),
            ProcessingInput(
                source= step_process_smallfleet.properties.ProcessingOutputConfig.Outputs[filename_hyperparameter_smallfleet].S3Output.S3Uri,
                destination="/opt/ml/processing/hyperparameter_tuning_smallfleet",
            ),
            ProcessingInput(
                source=step_process_smallfleet.properties.ProcessingOutputConfig.Outputs[
                    "feature_dict_smallfleet"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/feature_dict_smallfleet",
            ),
            ProcessingInput(
                source=step_process_smallfleet_train.properties.ProcessingOutputConfig.Outputs[
                    "smallfleet_model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/smallfleet_model",
            ),
    
        ],
        # outputs=[
        #     ProcessingOutput(output_name="small_df", source="/opt/ml/processing/small_df"),
        # ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/small_df",
                output_name="small_df",
                destination="s3://sagemaker-migration-hubble/model-output/small_df/",
                s3_upload_mode='EndOfJob',
                )
        ],
        code=os.path.join(BASE_DIR, "inference_smallfleet.py"),
#         code=os.path.join(BASE_DIR, "featureset"),
        # arguments=["--input-data", input_data],
    )
    step_process_smallfleet_inference = ProcessingStep(
        name="Inference_smallfleet",
        step_args=step_args_small_inference,
    )

    #################################################################################
    #combine all output, and write back to redshift
    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/hubble-save-output",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role='arn:aws:iam::318685047967:role/Role_Sagemaker',
        network_config=NetworkConfig(security_group_ids=['sg-0a9932248ac0ea55a', 'sg-0730ef060ad6ecf45'],
                                    subnets=['subnet-00e6236c99b7bce81','subnet-0287d47b678907dfd', 'subnet-0f422e73bc3febb31'])
                                    )
    step_args = script_processor.run(
        inputs=[
            ProcessingInput(
                source='s3://sagemaker-migration-hubble/model-output/express_df/',
                destination='/opt/ml/processing/express_df',
                s3_data_distribution_type='ShardedByS3Key',
                s3_input_mode='File',
                s3_data_type='S3Prefix'
            ),
            ProcessingInput(
                source='s3://sagemaker-migration-hubble/model-output/large_df/',
                destination='/opt/ml/processing/large_df',
                s3_data_distribution_type='ShardedByS3Key',
                s3_input_mode='File',
                s3_data_type='S3Prefix'
            ),
            ProcessingInput(
                source='s3://sagemaker-migration-hubble/model-output/small_df/',
                destination='/opt/ml/processing/small_df',
                s3_data_distribution_type='ShardedByS3Key',
                s3_input_mode='File',
                s3_data_type='S3Prefix'
            ),
               ],
        #inputs=[
        #    ProcessingInput(
        #        source=step_process_expresscheck_inference.properties.ProcessingOutputConfig.Outputs[
        #            "express_df"
        #        ].S3Output.S3Uri,
        #        destination="/opt/ml/processing/express_df",
        #    ),
        #    ProcessingInput(
        #        source=step_process_largefleet_inference.properties.ProcessingOutputConfig.Outputs[
        #            "large_df"
        #        ].S3Output.S3Uri,
        #        destination="/opt/ml/processing/large_df",
        #    ),
        #    ProcessingInput(
        #        source=step_process_smallfleet_inference.properties.ProcessingOutputConfig.Outputs[
        #            "small_df"
        #        ].S3Output.S3Uri,
        #        destination="/opt/ml/processing/small_df",
        #    ),
        #       ],
        outputs=[
            ProcessingOutput(output_name="output", source="/opt/ml/processing/output"),
        ],
        code=os.path.join(BASE_DIR, "write_to_redshift.py"),
    )
    step_data_save = ProcessingStep(
        name="Save-to-Redshift",
        step_args=step_args,
    )
    """
    
    ##########################################################################################
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
        ],
        steps=[step_data_extraction_train, step_data_extraction_test, # step_data_extraction_inference,
               step_process_train, step_process_test, # step_process_inference,
               step_process_largefleet, step_process_expresscheck, step_process_smallfleet,\
               step_process_expresscheck_train, step_process_largefleet_train, step_process_smallfleet_train,\
               # step_process_expresscheck_inference, step_process_largefleet_inference, step_process_smallfleet_inference, \
               # step_data_save
               ],
        sagemaker_session=pipeline_session,
    )
    return pipeline