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
    output_results: Output[HTML],
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
        
        # Generate HTML summary
        def generate_html_summary(scoring_response, config_path, eval_rows):
            html_content = []
            
            # Header
            html_content.append(f'<h1>Test Results</h1>')
            html_content.append(f'<p class="config-path"><strong>Config Path:</strong> <code>{config_path}</code></p>')
            
            for scoring_function_name, result in scoring_response.results.items():
                html_content.append(f'<h2>{scoring_function_name}</h2>')
                
                # Show aggregated results if available
                if hasattr(result, 'aggregated_results') and result.aggregated_results:
                    html_content.append('<div class="aggregated-results">')
                    html_content.append('<h3>Summary</h3>')
                    for metric_name, metric_data in result.aggregated_results.items():
                        if isinstance(metric_data, dict):
                            for key, value in metric_data.items():
                                html_content.append(f'<p><strong>{key}:</strong> {value}</p>')
                        else:
                            html_content.append(f'<p><strong>{metric_name}:</strong> {metric_data}</p>')
                    html_content.append('</div>')
                
                if hasattr(result, 'score_rows') and result.score_rows:
                    # Individual test results
                    html_content.append('<h3>Test Results</h3>')
                    html_content.append('<div class="test-results">')
                    
                    for i, (eval_row, score_row) in enumerate(zip(eval_rows, result.score_rows), 1):
                        html_content.append(f'<div class="test-result">')
                        html_content.append(f'<h4>Test {i}</h4>')
                        
                        # Input
                        html_content.append('<div class="test-section">')
                        html_content.append('<h5>Input</h5>')
                        html_content.append(f'<div class="content">{eval_row.get("input_query", "N/A")}</div>')
                        html_content.append('</div>')
                        
                        # Generated Answer
                        html_content.append('<div class="test-section">')
                        html_content.append('<h5>Generated Answer</h5>')
                        html_content.append(f'<div class="content">{eval_row.get("generated_answer", "N/A")}</div>')
                        html_content.append('</div>')
                        
                        # Expected Answer
                        html_content.append('<div class="test-section">')
                        html_content.append('<h5>Expected Answer</h5>')
                        html_content.append(f'<div class="content">{eval_row.get("expected_answer", "N/A")}</div>')
                        html_content.append('</div>')
                        
                        # Score
                        score = score_row.get('score', 'Unknown')
                        html_content.append('<div class="test-section">')
                        html_content.append('<h5>Score</h5>')
                        html_content.append(f'<div class="score-value">{score}</div>')
                        html_content.append('</div>')
                        
                        # Additional information (judge feedback, etc.)
                        additional_info = []
                        for key, value in score_row.items():
                            if key != 'score':
                                additional_info.append(f'<p><strong>{key}:</strong> {value}</p>')
                        
                        if additional_info:
                            html_content.append('<div class="test-section">')
                            html_content.append('<h5>Additional Information</h5>')
                            html_content.append('<div class="content">')
                            html_content.extend(additional_info)
                            html_content.append('</div>')
                            html_content.append('</div>')
                        
                        html_content.append('</div>')
                    
                    html_content.append('</div>')
                else:
                    html_content.append('<p class="no-results">No detailed results available.</p>')
            
            return '\n'.join(html_content)
        
        html_summary = generate_html_summary(scoring_response, config_path, eval_rows)
        
        # Write HTML summary to output
        with open(output_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results - {config_path}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin: 30px 0 20px 0;
            font-size: 1.8em;
        }}
        
        h3 {{
            color: #34495e;
            margin: 25px 0 15px 0;
            font-size: 1.4em;
        }}
        
        h4 {{
            color: #2c3e50;
            margin: 20px 0 10px 0;
            font-size: 1.2em;
        }}
        
        .config-path {{
            font-size: 1.1em;
            margin-bottom: 20px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
        }}
        
        .aggregated-results {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }}
        
        .aggregated-results p {{
            margin: 5px 0;
        }}
        
        .test-results {{
            margin-top: 20px;
        }}
        
        .test-result {{
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            background: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .test-section {{
            margin: 15px 0;
            padding: 10px 0;
            border-bottom: 1px solid #f1f3f4;
        }}
        
        .test-section:last-child {{
            border-bottom: none;
        }}
        
        .test-section h5 {{
            color: #495057;
            margin-bottom: 8px;
            font-size: 1em;
            font-weight: 600;
        }}
        
        .test-section .content {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #dee2e6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .score-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            padding: 8px 12px;
            background: #e9ecef;
            border-radius: 4px;
            display: inline-block;
            min-width: 60px;
            text-align: center;
        }}
        
        .no-results {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
            }}
            
            .header, .content {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .test-result {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Results</h1>
        </div>
        <div class="content">
            {html_summary}
        </div>
    </div>
</body>
</html>""")
        
        print(f"Processed llamastack test config: {config_path}")
        print(f"Generated markdown summary with {len(eval_rows)} test results")
        
        shutil.copy(output_path, output_results.path)


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