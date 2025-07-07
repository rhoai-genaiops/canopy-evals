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
        def generate_html_summary(scoring_response, config_path, eval_rows, scoring_params, config):
            html_content = []
            
            # Header
            html_content.append(f'<h1>Test Results</h1>')
            html_content.append(f'<p class="config-path"><strong>Config Path:</strong> <code>{config_path}</code></p>')
            
            # Test Configuration Details
            html_content.append('<div class="test-config">')
            html_content.append('<h3>Test Configuration</h3>')
            html_content.append('<div class="config-details">')
            
            if 'name' in config:
                html_content.append(f'<div class="config-detail">')
                html_content.append(f'<span class="detail-label">Name:</span>')
                html_content.append(f'<span class="detail-value">{config["name"]}</span>')
                html_content.append('</div>')
            
            if 'description' in config:
                html_content.append(f'<div class="config-detail">')
                html_content.append(f'<span class="detail-label">Description:</span>')
                html_content.append(f'<span class="detail-value">{config["description"]}</span>')
                html_content.append('</div>')
            
            if 'endpoint' in config:
                html_content.append(f'<div class="config-detail">')
                html_content.append(f'<span class="detail-label">Endpoint:</span>')
                html_content.append(f'<span class="detail-value endpoint">{config["endpoint"]}</span>')
                html_content.append('</div>')
            
            html_content.append('</div>')
            html_content.append('</div>')
            
            # Scoring Parameters
            html_content.append('<div class="scoring-params">')
            html_content.append('<h3>Scoring Configuration</h3>')
            
            for function_name, function_config in scoring_params.items():
                html_content.append(f'<div class="scoring-function">')
                html_content.append(f'<h4 class="function-name">{function_name}</h4>')
                
                if function_config is None:
                    html_content.append('<p class="null-config">No configuration (default behavior)</p>')
                else:
                    html_content.append('<div class="function-details">')
                    
                    for key, value in function_config.items():
                        html_content.append(f'<div class="config-item">')
                        html_content.append(f'<span class="config-key">{key}:</span>')
                        
                        if key == "prompt_template":
                            # Special formatting for prompt templates
                            html_content.append(f'<div class="prompt-template">{value}</div>')
                        elif isinstance(value, list):
                            # Format lists nicely
                            html_content.append('<ul class="config-list">')
                            for item in value:
                                html_content.append(f'<li>{item}</li>')
                            html_content.append('</ul>')
                        else:
                            # Regular values
                            html_content.append(f'<span class="config-value">{value}</span>')
                        
                        html_content.append('</div>')
                    
                    html_content.append('</div>')
                
                html_content.append('</div>')
            
            html_content.append('</div>')
            
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
                    
                    # Create single table for all test results
                    html_content.append('<table class="results-table">')
                    
                    # Create header row
                    html_content.append('<thead>')
                    html_content.append('<tr>')
                    html_content.append('<th>Test #</th>')
                    html_content.append('<th>Input</th>')
                    html_content.append('<th>Generated Answer</th>')
                    html_content.append('<th>Expected Answer</th>')
                    html_content.append('<th>Score</th>')
                    
                    # Get all additional column headers from first score row
                    if result.score_rows:
                        for key in result.score_rows[0].keys():
                            if key != 'score':
                                html_content.append(f'<th>{key.replace("_", " ").title()}</th>')
                    
                    html_content.append('</tr>')
                    html_content.append('</thead>')
                    
                    # Create data rows
                    html_content.append('<tbody>')
                    for i, (eval_row, score_row) in enumerate(zip(eval_rows, result.score_rows), 1):
                        html_content.append('<tr>')
                        html_content.append(f'<td class="test-number">{i}</td>')
                        html_content.append(f'<td class="content">{eval_row.get("input_query", "N/A")}</td>')
                        html_content.append(f'<td class="content">{eval_row.get("generated_answer", "N/A")}</td>')
                        html_content.append(f'<td class="content">{eval_row.get("expected_answer", "N/A")}</td>')
                        
                        score = score_row.get('score', 'Unknown')
                        html_content.append(f'<td class="score-value">{score}</td>')
                        
                        # Additional information columns
                        for key, value in score_row.items():
                            if key != 'score':
                                html_content.append(f'<td class="content">{value}</td>')
                        
                        html_content.append('</tr>')
                    
                    html_content.append('</tbody>')
                    html_content.append('</table>')
                    html_content.append('</div>')
                else:
                    html_content.append('<p class="no-results">No detailed results available.</p>')
            
            return '\n'.join(html_content)
        
        html_summary = generate_html_summary(scoring_response, config_path, eval_rows, scoring_params, config)
        
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
        
        .test-config {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }}
        
        .config-details {{
            margin-top: 15px;
        }}
        
        .config-detail {{
            display: flex;
            align-items: flex-start;
            margin: 10px 0;
            gap: 10px;
        }}
        
        .detail-label {{
            font-weight: 600;
            color: #495057;
            min-width: 100px;
            flex-shrink: 0;
        }}
        
        .detail-value {{
            color: #6c757d;
            flex-grow: 1;
        }}
        
        .detail-value.endpoint {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background: #ffffff;
            padding: 4px 8px;
            border-radius: 3px;
            border: 1px solid #dee2e6;
            color: #495057;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
        }}
        
        .scoring-params {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }}
        
        .scoring-function {{
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .function-name {{
            color: #007bff;
            margin: 0 0 15px 0;
            font-size: 1.1em;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 8px;
        }}
        
        .function-details {{
            margin-left: 10px;
        }}
        
        .config-item {{
            margin: 10px 0;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .config-key {{
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
        }}
        
        .config-value {{
            color: #6c757d;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.85em;
            background: #f8f9fa;
            padding: 4px 8px;
            border-radius: 3px;
            display: inline-block;
        }}
        
        .prompt-template {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 12px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.8em;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #495057;
            margin-top: 5px;
        }}
        
        .config-list {{
            margin: 5px 0 0 20px;
            padding: 0;
        }}
        
        .config-list li {{
            background: #f8f9fa;
            margin: 3px 0;
            padding: 4px 8px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.85em;
            color: #495057;
        }}
        
        .null-config {{
            color: #6c757d;
            font-style: italic;
            margin: 10px 0;
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
        
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .results-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            border-bottom: 2px solid #5a67d8;
        }}
        
        .results-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid #f1f3f4;
            vertical-align: top;
            max-width: 250px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        
        .results-table tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .results-table tbody tr:hover {{
            background: #e9ecef;
        }}
        
        .results-table td.test-number {{
            text-align: center;
            font-weight: bold;
            color: #2c3e50;
            width: 80px;
            min-width: 80px;
        }}
        
        .results-table td.content {{
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 0.9em;
        }}
        
        .results-table td.score-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            width: 100px;
            min-width: 100px;
        }}
        
        .results-table tbody tr:last-child td {{
            border-bottom: none;
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
            
            .results-table {{
                font-size: 0.8em;
            }}
            
            .results-table th,
            .results-table td {{
                padding: 8px 5px;
            }}
            
            .results-table td.content {{
                max-width: 150px;
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
        "base_url": "http://llama-stack.genaiops-rag.svc.cluster.local:80",
        "backend_url": "https://canopy-backend-user1-canopy.apps.cluster-gm86c.gm86c.sandbox1062.opentlc.com",
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