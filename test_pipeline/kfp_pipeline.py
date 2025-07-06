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
    Markdown
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
    
    # Scan for llamastack configs
    for path in glob.glob(os.path.join(base, "**/**_tests.yaml"), recursive=True):
        rel_path = os.path.relpath(path, base)
        llamastack_configs.append({"config_path": rel_path})

    from collections import namedtuple
    Output = namedtuple("Output", ["promptfoo_configs", "llamastack_configs"])
    return Output(promptfoo_configs=promptfoo_configs, llamastack_configs=llamastack_configs)



@component(base_image="python:3.11", packages_to_install=["git+https://github.com/meta-llama/llama-stack.git@release-0.2.12"])
def run_llamastack_tests_from_config(
    config_path: str,          # e.g., "Summary/Llama3.2-3b/summary_tests.yaml"
    repo_url: str,
    branch: str,
    base_url: str,
    backend_url: str,
    output_markdown: Output[Markdown],
):
    """Run llamastack-based tests from a summary_tests.yaml configuration file."""
    import os
    import subprocess
    import shutil
    import tempfile
    import yaml
    import json
    from llama_stack_client import LlamaStackClient
    from types import SimpleNamespace
    from urllib.parse import urljoin
    import requests

    lls_client = LlamaStackClient(
        base_url=base_url,
        timeout=600.0 # Default is 1 min which is far too little for some agentic tests, we set it to 10 min
    )

    def replace_txt_files(obj, base_path="."):
        if isinstance(obj, dict):
            return {k: replace_txt_files(v, base_path) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_txt_files(item, base_path) for item in obj]
        elif isinstance(obj, str) and obj.endswith(".txt"):
            file_path = os.path.join(base_path, obj)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return obj

    def send_request(payload, url):
        import httpx
        full_response = ""

        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json=payload) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[len("data: "):])
                            full_response += data.get("delta", "")
                        except json.JSONDecodeError:
                            continue

        return full_response

    def prompt_backend(prompt, backend_url, test_endpoint):
        url = urljoin(backend_url, test_endpoint)
        payload = {
            "prompt": prompt
        }
        return send_request(payload, url)


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

        test_endpoint = config["endpoint"]

        scoring_params = config["scoring_params"]
        scoring_params = replace_txt_files(scoring_params, os.path.dirname(full_config_path))

        eval_rows = []
        for test in config["tests"]:
            if "dataset" in test:
                pass
            else:
                prompt = test["prompt"]
                eval_rows.append(
                    {
                        "input_query": prompt,
                        "generated_answer": prompt_backend(prompt, backend_url, test_endpoint),
                        "expected_answer": test["expected_result"],
                    }
                )

        scoring_response = lls_client.scoring.score(
            input_rows=eval_rows, scoring_functions=scoring_params
        )
        
        # Generate markdown summary
        def generate_markdown_summary(scoring_response, config_path):
            markdown_lines = []
            markdown_lines.append(f"# Test Results Summary")
            markdown_lines.append(f"**Config Path:** `{config_path}`")
            markdown_lines.append("")
            
            for scoring_function_name, result in scoring_response.results.items():
                markdown_lines.append(f"## {scoring_function_name}")
                markdown_lines.append("")
                
                if hasattr(result, 'score_rows') and result.score_rows:
                    # Count scores
                    score_counts = {}
                    for row in result.score_rows:
                        score = row.get('score', 'Unknown')
                        score_counts[score] = score_counts.get(score, 0) + 1
                    
                    # Add summary statistics
                    total_tests = len(result.score_rows)
                    markdown_lines.append(f"**Total Tests:** {total_tests}")
                    markdown_lines.append("")
                    
                    # Score distribution
                    markdown_lines.append("### Score Distribution")
                    for score, count in sorted(score_counts.items()):
                        percentage = (count / total_tests) * 100
                        markdown_lines.append(f"- **{score}**: {count} ({percentage:.1f}%)")
                    markdown_lines.append("")
                    
                    # Detailed results
                    markdown_lines.append("### Detailed Results")
                    for i, row in enumerate(result.score_rows, 1):
                        score = row.get('score', 'Unknown')
                        feedback = row.get('judge_feedback', 'No feedback provided')
                        markdown_lines.append(f"#### Test {i}")
                        markdown_lines.append(f"**Score:** {score}")
                        markdown_lines.append(f"**Feedback:** {feedback}")
                        markdown_lines.append("")
                else:
                    markdown_lines.append("No detailed results available.")
                    markdown_lines.append("")
            
            return "\n".join(markdown_lines)
        
        markdown_summary = generate_markdown_summary(scoring_response, config_path)
        
        # Write markdown summary to HTML output
        with open(output_path, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #ddd; }}
        h2 {{ border-bottom: 1px solid #eee; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .score {{ font-weight: bold; color: #007acc; }}
        .feedback {{ margin-left: 20px; font-style: italic; color: #666; }}
    </style>
</head>
<body>
    <div id="markdown-content">
        {markdown_summary.replace(chr(10), '<br>').replace('**', '<strong>').replace('**', '</strong>').replace('`', '<code>').replace('`', '</code>').replace('# ', '<h1>').replace('</h1><br>', '</h1>').replace('## ', '<h2>').replace('</h2><br>', '</h2>').replace('### ', '<h3>').replace('</h3><br>', '</h3>').replace('#### ', '<h4>').replace('</h4><br>', '</h4>').replace('- ', '<li>').replace('<li>', '<ul><li>').replace('</li><br><li>', '</li><li>').replace('</li><br><br>', '</li></ul><br>')}
    </div>
</body>
</html>
            """)
        
        print(f"Processed llamastack test config: {config_path}")
        print(f"Generated markdown summary with {len(eval_rows)} test results")
        
        shutil.copy(output_path, output_markdown.path)
        
        return markdown_summary


@dsl.pipeline(
    name="Promptfoo Testing Pipeline",
    description="Pipeline for running promptfoo tests across repositories"
)
def promptfoo_test_pipeline(
    repo_url: str,
    branch: str = "main",
    workspace_pvc: str = "canopy-workspace-pvc",
    base_url: str = "",
    backend_url: str = "",
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

    # Step 4: Run llamastack tests for each llamastack config
    with dsl.ParallelFor(scan_task.outputs["llamastack_configs"]) as config:
        run_llamastack_tests_from_config(
            config_path=config.config_path,
            repo_url=repo_url,
            branch=branch,
            base_url=base_url,
            backend_url=backend_url,
        )


if __name__ == '__main__':
    arguments = {
        "repo_url": "https://github.com/rhoai-genaiops/canopy-prompts",
        "branch": "main",
        "workspace_pvc": "canopy-workspace-pvc",
        "base_url": "http://llamastack-server-genaiops-playground.apps.dev.rhoai.rh-aiservices-bu.com",
        "backend_url": "https://canopy-backend-genaiops-playground.apps.dev.rhoai.rh-aiservices-bu.com",
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