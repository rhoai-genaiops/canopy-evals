"""
Kubeflow Pipeline for Automated Promptfoo Testing

This pipeline implements an automated testing workflow that:
1. Clones a git repository
2. Scans directories for promptfoo configuration
3. Executes promptfoo tests
"""

from typing import NamedTuple, Optional, List
from kfp import dsl, components
from kfp.components import InputPath, OutputPath

# Define the container image from our Containerfile
CONTAINER_IMAGE = "promptfoo-test-runner:latest"

@components.create_component_from_func
def git_clone_op(
    repo_url: str,
    dest_path: OutputPath("Directory"),
    branch: str = "main"
) -> NamedTuple("Outputs", [("repo_path", str)]):
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
        dest_path
    ], check=True)
    
    return (dest_path,)

@components.create_component_from_func
def scan_directory_op(
    base_path: InputPath("Directory"),
    pattern: str = "**/promptfooconfig.yaml"
) -> NamedTuple("Outputs", [("config_paths", List[str])]):
    """Scan directory for promptfoo configuration files.
    
    Args:
        base_path: Base directory to start scanning from
        pattern: Glob pattern to match config files
    
    Returns:
        NamedTuple containing list of found configuration file paths
    """
    import glob
    import os
    from collections import namedtuple
    
    # Find all promptfoo config files
    config_files = glob.glob(os.path.join(base_path, pattern), recursive=True)
    
    # Create named tuple for output
    outputs = namedtuple("Outputs", ["config_paths"])
    return outputs(config_paths=config_files)

@dsl.component(
    base_image=CONTAINER_IMAGE,
    packages_to_install=["pyyaml"]
)
def run_promptfoo_tests(
    config_path: str,
    output_file: OutputPath("JsonFile"),
    timeout: int = 3600
) -> NamedTuple("Outputs", [
    ("passed", bool),
    ("total_tests", int),
    ("failed_tests", int)
]):
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
    
    try:
        # Run promptfoo tests
        result = subprocess.run(
            ["promptfoo", "eval", "--config", config_path, "--output", output_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse results
        with open(output_file, 'r') as f:
            test_results = json.load(f)
        
        total_tests = len(test_results.get("results", []))
        failed_tests = sum(1 for r in test_results.get("results", []) 
                          if not r.get("pass", False))
        passed = failed_tests == 0
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Tests timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Test execution failed: {str(e)}")
    
    # Create named tuple for output
    outputs = namedtuple("Outputs", ["passed", "total_tests", "failed_tests"])
    return outputs(passed=passed, total_tests=total_tests, failed_tests=failed_tests)

@dsl.pipeline(
    name="Promptfoo Testing Pipeline",
    description="Pipeline for running promptfoo tests across repositories"
)
def promptfoo_test_pipeline(
    repo_url: str,
    branch: str = "main",
    test_timeout: int = 3600
):
    """Pipeline that clones a repository and runs promptfoo tests.
    
    Args:
        repo_url: URL of the git repository to test
        branch: Git branch to test (default: main)
        test_timeout: Maximum time in seconds for test execution
    """
    
    # Clone repository
    clone_task = git_clone_op(repo_url=repo_url, branch=branch)
    
    # Scan for config files
    scan_task = scan_directory_op(
        base_path=clone_task.outputs["repo_path"]
    )
    
    # Run tests for each config file found
    with dsl.ParallelFor(scan_task.outputs["config_paths"]) as config_path:
        test_task = run_promptfoo_tests(
            config_path=config_path,
            timeout=test_timeout
        )

if __name__ == "__main__":
    # Compile the pipeline
    from kfp.compiler import Compiler
    Compiler().compile(
        pipeline_func=promptfoo_test_pipeline,
        package_path="promptfoo_test_pipeline.yaml"
    )