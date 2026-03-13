import os

from celery import Celery
from celery.signals import worker_process_init
from dotenv import load_dotenv

load_dotenv()

app = Celery("sambandh")
app.conf.update(
    broker_url=os.environ.get("REDIS_URL", "redis://localhost:6380/0"),
    result_backend=None,
    task_ignore_result=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    worker_max_tasks_per_child=50,
    task_default_queue="ocr",
    broker_transport_options={"visibility_timeout": 3600},
    broker_connection_retry_on_startup=True,
)

app.autodiscover_tasks(["sambandh.tasks"])


@worker_process_init.connect
def init_worker_process(**kwargs):
    """Called once when each prefork child process starts.

    Initializes the Docling converter so it is reused across all tasks
    handled by this child process.
    """
    from sambandh.converter import warm_converter

    warm_converter()
