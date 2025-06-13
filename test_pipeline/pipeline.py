"""
Kubeflow Pipeline for Automated Promptfoo Testing

This pipeline implements an automated testing workflow that:
1. Clones a git repository
2. Scans directories for promptfoo configuration
3. Executes promptfoo tests
"""

import kfp
from typing import NamedTuple, Optional, List
from kfp import dsl
from kfp.dsl import (
    component,
    Output,
    HTML,
)
from kfp import kubernetes
from typing import NamedTuple, List
from dataclasses import dataclass

@dataclass
class ConfigPair:
    config_path: str
    result_path: str

# Define the container image from our Containerfile
CONTAINER_IMAGE = "quay.io/rlundber/promptfoo:0.1"

@component(base_image='python:3.9')
def git_clone_op(
    repo_url: str,
    branch: str = "main"
):
    """Clone a Git repository to the specified path.
    
    Args:
        repo_url: URL of the git repository to clone
        branch: Branch to checkout (default: main)
        dest_path: Output path where repository will be cloned
    
    Returns:
        NamedTuple containing the path to the cloned repository
    """
    import os
    import subprocess
    
    # Clone the repository
    subprocess.run([
        "git", "clone",
        "--branch", branch,
        "--single-branch",
        "--depth", "1",
        repo_url,
        "/prompts"
    ], check=True)

    for item in os.listdir("/prompts"):
        print(item)

@component(base_image='python:3.9')
def scan_directory_op(
    pattern: str = "**/promptfooconfig.yaml"
) -> List:
    import glob
    import os
    from collections import namedtuple

    config_files = glob.glob(os.path.join("/prompts", pattern), recursive=True)
    pairs = [{"config_path": c, "result_path": os.path.join(os.path.dirname(c), "result.html")} for c in config_files]

    return pairs

@dsl.container_component
def run_promptfoo_tests(config_path: str, output_path: str):
    return dsl.ContainerSpec(image=CONTAINER_IMAGE, command=["promptfoo", "eval", "--config", config_path, "--output", output_path])#, args=[name])


@component(base_image='python:3.9', packages_to_install=["pyyaml"])
def process_test_output(
    result_path: str,
    test_results: Output[HTML],
):
    """Execute promptfoo tests for a given configuration.
    
    Args:
        config_path: Path to promptfoo configuration file
        output_file: Path where test results will be saved
        timeout: Maximum time in seconds to wait for tests
    
    Returns:
        NamedTuple containing test execution results
    """
    import json
    import subprocess
    import os
    from collections import namedtuple

    # Run promptfoo tests
    with open(result_path, 'r') as f:
        test_results_content = result_path.read()

    with open(test_results.path, 'w') as dest:
        dest.write(test_results_content)

@dsl.pipeline(
    name="Promptfoo Testing Pipeline",
    description="Pipeline for running promptfoo tests across repositories"
)
def promptfoo_test_pipeline(
    repo_url: str,
    branch: str = "main",
    workspace_pvc: str = "canopy-workspace-pvc",
):
    import os
    """Pipeline that clones a repository and runs promptfoo tests.
    
    Args:
        repo_url: URL of the git repository to test
        branch: Git branch to test (default: main)
        test_timeout: Maximum time in seconds for test execution
    """
    
    # Clone repository
    clone_task = git_clone_op(repo_url=repo_url, branch=branch)
    kubernetes.mount_pvc(
        clone_task,
        pvc_name=workspace_pvc,
        mount_path='/prompts',
    )
    
    # Scan for config files
    scan_task = scan_directory_op()
    scan_task.after(clone_task)
    kubernetes.mount_pvc(
        scan_task,
        pvc_name=workspace_pvc,
        mount_path='/prompts',
    )
    
    # Run tests for each config file found
    with dsl.ParallelFor(scan_task.output) as pair:
        test_task = run_promptfoo_tests(
            config_path=pair.config_path,
            output_path=pair.result_path
        )

        kubernetes.mount_pvc(
            test_task,
            pvc_name=workspace_pvc,
            mount_path='/prompts',
        )

        process_output_task = process_test_output(
            result_path=pair.result_path
        )
        process_output_task.after(test_task)

        kubernetes.mount_pvc(
            process_output_task,
            pvc_name=workspace_pvc,
            mount_path='/prompts',
        )


if __name__ == '__main__':
    arguments = {
        "repo_url": "",
        "branch": "main",
        "workspace_pvc": "canopy-workspace-pvc",
    }
        
    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint =\
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert =\
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        promptfoo_test_pipeline,
        arguments=arguments,
        experiment_name="kfp-training-pipeline",
        enable_caching=True
    )