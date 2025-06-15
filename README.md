# Automated LLM Testing with Kubeflow and promptfoo on OpenShift

This guide demonstrates how to implement automated testing for Large Language Models (LLMs) and AI workflows using Kubeflow pipelines integrated with promptfoo on Red Hat OpenShift. The solution enables scalable, reproducible testing of LLM behaviors across multiple test scenarios.

## Overview

Testing LLMs and AI workflows presents unique challenges due to their complex nature and potentially non-deterministic outputs. This implementation combines Kubeflow's powerful orchestration capabilities with promptfoo's specialized LLM testing framework to create an automated, scalable testing pipeline.

Key features:
- Automated discovery and execution of promptfoo test configurations
- Parallel test execution in isolated environments
- Distributed testing support
- Integrated HTML report generation
- Persistent storage for test artifacts

## Prerequisites

- Red Hat OpenShift cluster with OpenShift AI/Data Science components installed
- Kubeflow Pipelines operator
- Access to container registry (we use quay.io)
- Git repository containing promptfoo test configurations

## Architecture

The testing pipeline is implemented as a Kubeflow workflow with three main components:

1. **Repository Management**
   - Git clone operation for fetching test configurations
   - Workspace initialization using PersistentVolumeClaim

2. **Test Discovery**
   - Recursive scanner for promptfoo configurations
   - Automatic validation of test specifications

3. **Test Execution**
   - Parallel test runner using custom promptfoo container
   - Isolated test environments
   - Centralized result collection

## Implementation Details

### Storage Configuration

The pipeline uses a CephFS-backed PersistentVolumeClaim for workspace management:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: promptfoo-workspace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  storageClassName: cephfs
```

### Pipeline Components

The Kubeflow pipeline is structured using the Python DSL:

```python
@dsl.pipeline(
    name="LLM Testing Pipeline",
    description="Automated LLM testing using promptfoo"
)
def llm_test_pipeline(repo_url: str):
    workspace_vol = dsl.PipelineVolume(pvc="promptfoo-workspace")
    
    # Clone repository
    clone_step = dsl.ContainerOp(
        name="clone-repo",
        image="alpine/git:latest",
        command=["git", "clone", repo_url, "/workspace"],
        pvolumes={"/workspace": workspace_vol}
    )
    
    # Scan for test configs
    scan_step = dsl.ContainerOp(
        name="scan-configs",
        image="quay.io/rlundber/promptfoo:0.4",
        command=["find", "/workspace", "-name", "promptfooconfig.yaml"],
        pvolumes={"/workspace": workspace_vol}
    )
    
    # Execute tests in parallel
    with dsl.ParallelFor(scan_step.output) as config:
        test_step = dsl.ContainerOp(
            name="run-tests",
            image="quay.io/rlundber/promptfoo:0.4",
            command=["promptfoo", "eval", "-c", config],
            pvolumes={"/workspace": workspace_vol}
        )
```

### Test Execution

The pipeline uses a custom promptfoo container (`quay.io/rlundber/promptfoo:0.4`) that provides:
- Isolated test environment
- Built-in promptfoo CLI
- Result aggregation capabilities

## Usage

1. Create the PersistentVolumeClaim:
```bash
oc apply -f test_pipeline/pvc.yaml
```

2. Configure your test repository with promptfoo configurations.

3. Submit the pipeline:
```bash
python test_pipeline/kfp_pipeline.py \
  --repo-url https://github.com/your/test-repo.git
```

4. Monitor the pipeline execution in the Kubeflow dashboard.

## Benefits

This implementation provides several advantages:

- **Scalability**: Parallel test execution with resource isolation
- **Reproducibility**: Consistent test environments through containerization
- **Flexibility**: Support for various LLM providers and test scenarios
- **Integration**: Seamless integration with OpenShift's enterprise features
- **Observability**: Comprehensive test reporting and result tracking

## Conclusion

By combining Kubeflow pipelines with promptfoo on OpenShift, we've created a robust solution for automated LLM testing. This approach enables teams to maintain quality and reliability in their AI applications through systematic testing and validation.

The implementation demonstrates the power of OpenShift's container orchestration capabilities when combined with specialized AI testing tools, providing a foundation for scalable and reliable AI application development.
