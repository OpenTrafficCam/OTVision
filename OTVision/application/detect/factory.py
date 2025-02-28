from OTVision.config import DetectConfig
from OTVision.domain.object_detection import ObjectDetector, ObjectDetectorFactory


class ObjectDetectorCachedFactory(ObjectDetectorFactory):

    def __init__(self, other: ObjectDetectorFactory) -> None:
        self._other = other
        self.__cache: dict[str, ObjectDetector] = {}

    def create(self, config: DetectConfig) -> ObjectDetector:
        if cached_model := self.__cache.get(config.yolo_config.weights):
            return cached_model
        model = self._other.create(config)
        self.__add_to_cache(model)
        return model

    def __add_to_cache(self, model: ObjectDetector) -> None:
        weights = model.config.yolo_config.weights
        if not self.__cache.get(weights):
            self.__cache[weights] = model

    def __remove_from_cache(self, weights: str) -> None:
        if self.__cache.get(weights):
            del self.__cache[weights]
