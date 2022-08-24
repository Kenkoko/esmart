import concurrent.futures
import copy
import gc
import os
from cmath import nan

from esmart.job import Job
from esmart.job.trace import Trace
from esmart.job.train_normal import TrainingJobNormal
from esmart.misc import init_from
from esmart.util.io import get_checkpoint_file, load_checkpoint
from esmart.util.metric import Metric

class SearchJob(Job):
    """Base class of jobs for hyperparameter search.
    Provides functionality for scheduling training jobs across workers.
    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        self.ready_task_results = list()  #: set of results
        self.on_error = self.config.check("search.on_error", ["abort", "continue"])
        if self.__class__ == SearchJob:
            for f in Job.job_created_hooks:
                f(self)
        self.running_tasks = set()  #: set of futures currently runnning

    @staticmethod
    def create(config, dataset, parent_job=None):
        """Factory method to create a search job."""

        search_type = config.get("search.type")
        class_name = config.get_default(f"{search_type}.class_name")
        return init_from(class_name, config.modules(), config, dataset, parent_job)

    def submit_task(self, task, task_arg, wait_when_full=True):
        """Runs the given task with the given argument.
        """
        self.ready_task_results.append(task(task_arg))
    
    def wait_task(self, return_when=concurrent.futures.FIRST_COMPLETED):
        """Waits for one or more running tasks to complete.
        Results of all completed tasks are copied into ``self.ready_task_results``.
        When no task is running, does nothing.
        """
        if len(self.running_tasks) > 0:
            self.config.log("Waiting for tasks to complete...")
            ready_tasks, self.running_tasks = concurrent.futures.wait(
                self.running_tasks, return_when=return_when
            )
            for task in ready_tasks:
                self.ready_task_results.append(task.result())

def _run_train_job(sicnk, device=None):
    """Runs a training job and returns the trace entry of its best validation result.
    Also takes are of appropriate tracing.
    """

    search_job, train_job_index, train_job_config, train_job_count, trace_keys = sicnk

    try:
        search_job.config.log(
            "Starting training job {} ({}/{}) on device {}...".format(
                train_job_config.folder,
                train_job_index + 1,
                train_job_count,
                train_job_config.get("job.device"),
            )
        )

        checkpoint_file = get_checkpoint_file(train_job_config)
        if checkpoint_file is not None:
            checkpoint = load_checkpoint(
                checkpoint_file, train_job_config.get("job.device")
            )
            job = Job.create_from(
                checkpoint=checkpoint,
                new_config=train_job_config,
                dataset=search_job.dataset,
                parent_job=search_job,
            )
        else:
            job = Job.create(
                config=train_job_config,
                dataset=search_job.dataset,
                parent_job=search_job,
            )
        # process the trace entries to far (in case of a resumed job)
        metric_name = search_job.config.get("valid.metric")
        valid_trace = []

        def copy_to_search_trace(job, trace_entry=None):
            if trace_entry is None:
                trace_entry = job.valid_trace[-1]
            trace_entry = copy.deepcopy(trace_entry)
            for key in trace_keys:
                # Process deprecated options to some extent. Support key renames, but
                # not value renames.
                actual_key = {key: None}
                if len(actual_key) > 1:
                    raise KeyError(
                        f"{key} is deprecated but cannot be handled automatically"
                    )
                actual_key = next(iter(actual_key.keys()))
                value = train_job_config.get(actual_key)
                trace_entry[key] = value

            trace_entry["folder"] = os.path.split(train_job_config.folder)[1]
            metric_value = Trace.get_metric(trace_entry, metric_name)
            trace_entry["metric_name"] = metric_name
            trace_entry["metric_value"] = metric_value
            trace_entry["parent_job_id"] = search_job.job_id
            search_job.config.trace(**trace_entry)
            valid_trace.append(trace_entry)

        for trace_entry in job.valid_trace:
            copy_to_search_trace(None, trace_entry)

        job.post_valid_hooks.append(copy_to_search_trace)
        job.run()

        search_job.config.log("Best result in this training job:")
        best = None
        best_metric = None
        for trace_entry in valid_trace:
            metric = trace_entry["metric_value"]
            if not best or Metric(search_job).better(metric, best_metric):
                best = trace_entry
                best_metric = metric

        # record the best result of this job
        best["child_job_id"] = best["job_id"]
        for k in ["job", "job_id", "type", "parent_job_id", "scope", "event"]:
            if k in best:
                del best[k]
        search_job.trace(
            event="search_completed",
            echo=True,
            echo_prefix="  ",
            log=True,
            scope="train",
            **best,
        )

        del job
        gc.collect()

        return (train_job_index, best, best_metric)
    except BaseException as e:
        search_job.config.log(
            "Trial {:05d} failed: {}".format(train_job_index, repr(e))
        )
        if search_job.on_error == "continue":
            return (train_job_index, None, None)
        else:
            search_job.config.log(
                "Aborting search due to failure of trial {:05d}".format(train_job_index)
            )
            raise e
