# TODO: Might need to change this
from .convert.convert import main as convert
from .detect.detect import main as detect
from .track.track import main as track
from .view.view import main as view

__all__ = [detect, track, convert, view]
