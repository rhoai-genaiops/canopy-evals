"""
Kubeflow Pipeline for Automated Promptfoo Testing

This pipeline implements an automated testing workflow that:
1. Clones a git repository
2. Scans directories for promptfoo configuration
3. Executes promptfoo tests
"""

import kfp
from typing import NamedTuple, Optional, List
from kfp import dsl, components
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Artifact,
    HTML,
)
from kfp import kubernetes
from typing import NamedTuple, List
from dataclasses import dataclass

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
    import shutil

    folder = "/prompts"

    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    
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

@component(base_image="python:3.9")
def scan_directory_op() -> NamedTuple("Output", [("promptfoo_configs", List[dict]), ("llamastack_configs", List[dict])]):
    import glob
    import os

    promptfoo_configs = []
    llamastack_configs = []
    base = "/prompts"
    
    # Scan for promptfoo configs
    for path in glob.glob(os.path.join(base, "**/promptfooconfig.yaml"), recursive=True):
        rel_path = os.path.relpath(path, base)
        promptfoo_configs.append({"config_path": rel_path})
    
    # Scan for llamastack configs
    for path in glob.glob(os.path.join(base, "**/summary_tests.yaml"), recursive=True):
        rel_path = os.path.relpath(path, base)
        llamastack_configs.append({"config_path": rel_path})

    from collections import namedtuple
    Output = namedtuple("Output", ["promptfoo_configs", "llamastack_configs"])
    return Output(promptfoo_configs=promptfoo_configs, llamastack_configs=llamastack_configs)



@component(base_image="quay.io/rlundber/promptfoo:0.4")
def run_promptfoo_tests_from_config(
    config_path: str,          # e.g., "tests/foo/promptfooconfig.yaml"
    repo_url: str,
    branch: str,
    output_html: Output[HTML],
):
    import os
    import subprocess
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "repo")

        # Clone the repo
        subprocess.run([
            "git", "clone", "--branch", branch,
            "--single-branch", "--depth", "1",
            repo_url, repo_dir
        ], check=True)

        full_config_path = os.path.join(repo_dir, config_path)
        output_path = os.path.join(tmpdir, "result.html")

        # Run promptfoo
        result = subprocess.run(
            ["promptfoo", "eval", "--config", full_config_path, "--output", output_path],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": "/tmp"},
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        shutil.copy(output_path, output_html.path)


@component(base_image="python:3.9")
def run_llamastack_tests_from_config(
    config_path: str,          # e.g., "Summary/Llama3.2-3b/summary_tests.yaml"
    repo_url: str,
    branch: str,
    output_html: Output[HTML],
):
    """Run llamastack-based tests from a summary_tests.yaml configuration file."""
    import os
    import subprocess
    import shutil
    import tempfile
    import yaml
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "repo")

        # Clone the repo
        subprocess.run([
            "git", "clone", "--branch", branch,
            "--single-branch", "--depth", "1",
            repo_url, repo_dir
        ], check=True)

        full_config_path = os.path.join(repo_dir, config_path)
        output_path = os.path.join(tmpdir, "result.html")

        # Load the summary_tests.yaml configuration
        with open(full_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # For now, create a placeholder HTML output with config details
        # This can be extended to integrate with actual llamastack evaluation
        with open(output_path, "w") as f:
            f.write(f"""<html>
            <head><title>Llamastack Test Results</title></head>
            <body>
                <h1>Llamastack Test Results</h1>
                <h2>Configuration: {config_path}</h2>
                <h3>Test Details:</h3>
                <ul>
                    <li><strong>Name:</strong> {config.get('name', 'N/A')}</li>
                    <li><strong>Description:</strong> {config.get('description', 'N/A')}</li>
                    <li><strong>Model:</strong> {config.get('model', 'N/A')}</li>
                    <li><strong>Endpoint:</strong> {config.get('endpoint', 'N/A')}</li>
                    <li><strong>Number of Tests:</strong> {len(config.get('tests', []))}</li>
                </ul>
                <h3>Tests:</h3>
                <table border="1">
                    <tr><th>Test Text</th><th>Expected Result</th></tr>
            """)
            
            for test in config.get('tests', []):
                f.write(f"""
                    <tr>
                        <td>{test.get('text', 'N/A')}</td>
                        <td>{test.get('expected_result', 'N/A')}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
                <p><em>Note: This is a placeholder implementation. 
                Actual llamastack evaluation integration to be implemented.</em></p>
            </body>
            </html>""")

        print(f"Processed llamastack test config: {config_path}")
        print(f"Config details: {json.dumps(config, indent=2)}")
        
        shutil.copy(output_path, output_html.path)


@dsl.pipeline(
    name="Promptfoo Testing Pipeline",
    description="Pipeline for running promptfoo tests across repositories"
)
def promptfoo_test_pipeline(
    repo_url: str,
    branch: str = "main",
    workspace_pvc: str = "canopy-workspace-pvc",
):
    # Step 1: Clone repo
    clone_task = git_clone_op(repo_url=repo_url, branch=branch)
    kubernetes.mount_pvc(
        clone_task,
        pvc_name=workspace_pvc,
        mount_path='/prompts',
    )

    # Step 2: Scan for all test configs
    scan_task = scan_directory_op()
    scan_task.after(clone_task)
    kubernetes.mount_pvc(
        scan_task,
        pvc_name=workspace_pvc,
        mount_path='/prompts',
    )

    # Step 3: Run promptfoo tests for each promptfoo config
    with dsl.ParallelFor(scan_task.outputs["promptfoo_configs"]) as config:
        run_promptfoo_tests_from_config(
            config_path=config.config_path,
            repo_url=repo_url,
            branch=branch
        )

    # Step 4: Run llamastack tests for each llamastack config
    with dsl.ParallelFor(scan_task.outputs["llamastack_configs"]) as config:
        run_llamastack_tests_from_config(
            config_path=config.config_path,
            repo_url=repo_url,
            branch=branch
        )


if __name__ == '__main__':
    arguments = {
        "repo_url": "https://github.com/rhoai-genaiops/canopy-prompts",
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
        enable_caching=False
    )