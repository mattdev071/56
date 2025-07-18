import queue
import threading
import os
from uuid import UUID
from typing import Dict, List, Optional
from dataclasses import dataclass

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_container
from miner.logic.job_handler import start_tuning_container_diffusion


logger = get_logger(__name__)


@dataclass
class GPUResource:
    gpu_id: int
    is_available: bool = True
    current_job: Optional[str] = None
    memory_usage: float = 0.0


class AdvancedTrainingWorker:
    def __init__(self):
        logger.info("=" * 80)
        logger.info("STARTING ADVANCED TRAINING WORKER WITH H100 PARALLEL PROCESSING")
        logger.info("=" * 80)

        # Initialize GPU resources
        self.gpu_resources = self._initialize_gpu_resources()
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        self.active_jobs: Dict[str, List[int]] = {}  # job_id -> list of gpu_ids
        self.job_threads: Dict[str, threading.Thread] = {}
        
        # Start main worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.docker_client = docker.from_env()
        
        logger.info(f"Initialized with {len(self.gpu_resources)} GPU resources")

    def _initialize_gpu_resources(self) -> Dict[int, GPUResource]:
        """Initialize GPU resources based on available H100s"""
        gpu_ids = [int(id) for id in os.getenv("GPU_IDS", "0,1,2,3").split(",")]
        return {gpu_id: GPUResource(gpu_id=gpu_id) for gpu_id in gpu_ids}

    def _get_available_gpus(self, required_gpus: int = 1) -> List[int]:
        """Get available GPUs for job execution"""
        available_gpus = [gpu_id for gpu_id, resource in self.gpu_resources.items() 
                         if resource.is_available]
        
        if len(available_gpus) >= required_gpus:
            return available_gpus[:required_gpus]
        return []

    def _allocate_gpus(self, job_id: str, gpu_ids: List[int]):
        """Allocate GPUs to a job"""
        for gpu_id in gpu_ids:
            self.gpu_resources[gpu_id].is_available = False
            self.gpu_resources[gpu_id].current_job = job_id
        self.active_jobs[job_id] = gpu_ids
        logger.info(f"Allocated GPUs {gpu_ids} to job {job_id}")

    def _release_gpus(self, job_id: str):
        """Release GPUs from a completed job"""
        if job_id in self.active_jobs:
            gpu_ids = self.active_jobs[job_id]
            for gpu_id in gpu_ids:
                self.gpu_resources[gpu_id].is_available = True
                self.gpu_resources[gpu_id].current_job = None
            del self.active_jobs[job_id]
            logger.info(f"Released GPUs {gpu_ids} from job {job_id}")

    def _calculate_required_gpus(self, job: Job) -> int:
        """Calculate required GPUs based on model size and task type"""
        if isinstance(job, TextJob):
            # For text models, use more GPUs for larger models
            model_size = self._estimate_model_size(job.model)
            if model_size > 70:  # 70B+ models
                return 4
            elif model_size > 35:  # 35B+ models
                return 2
            else:
                return 1
        elif isinstance(job, DiffusionJob):
            # Diffusion models typically need 2-4 GPUs
            return 2
        return 1

    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in billions of parameters"""
        if "70b" in model_name.lower() or "70b" in model_name:
            return 70
        elif "34b" in model_name.lower() or "34b" in model_name:
            return 34
        elif "13b" in model_name.lower() or "13b" in model_name:
            return 13
        elif "7b" in model_name.lower() or "7b" in model_name:
            return 7
        elif "3b" in model_name.lower() or "3b" in model_name:
            return 3
        return 7  # Default assumption

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            
            try:
                required_gpus = self._calculate_required_gpus(job)
                available_gpus = self._get_available_gpus(required_gpus)
                
                if len(available_gpus) < required_gpus:
                    logger.warning(f"Not enough GPUs for job {job.job_id}. Required: {required_gpus}, Available: {len(available_gpus)}")
                    # Re-queue the job for later processing
                    self.job_queue.put(job)
                    continue
                
                # Allocate GPUs and start job
                self._allocate_gpus(job.job_id, available_gpus)
                
                # Start job in separate thread
                job_thread = threading.Thread(
                    target=self._process_job,
                    args=(job, available_gpus),
                    daemon=True
                )
                self.job_threads[job.job_id] = job_thread
                job_thread.start()
                
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                self._release_gpus(job.job_id)
            finally:
                self.job_queue.task_done()

    def _process_job(self, job: Job, gpu_ids: List[int]):
        """Process a job with allocated GPUs"""
        try:
            logger.info(f"Processing job {job.job_id} on GPUs {gpu_ids}")
            
            # Set environment variables for multi-GPU training
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            os.environ["NVIDIA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            
            if isinstance(job, TextJob):
                start_tuning_container(job, gpu_ids)
            elif isinstance(job, DiffusionJob):
                start_tuning_container_diffusion(job, gpu_ids)
            
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
        finally:
            self._release_gpus(job.job_id)
            if job.job_id in self.job_threads:
                del self.job_threads[job.job_id]

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job
        logger.info(f"Enqueued job {job.job_id}")

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND

    def get_gpu_utilization(self) -> Dict[int, float]:
        """Get current GPU utilization"""
        return {gpu_id: resource.memory_usage for gpu_id, resource in self.gpu_resources.items()}

    def shutdown(self):
        # Stop all active job threads
        for thread in self.job_threads.values():
            thread.join(timeout=5)
        
        self.thread.join()
        self.docker_client.close()


# Backward compatibility
class TrainingWorker(AdvancedTrainingWorker):
    pass
