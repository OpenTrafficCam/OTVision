import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(message)s", level=logging.INFO
)

log = logging.getLogger(__name__)
