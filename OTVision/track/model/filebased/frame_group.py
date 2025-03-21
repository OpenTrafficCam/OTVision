from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def get_output_file(file: Path, with_suffix: str) -> Path:
    return file.with_suffix(with_suffix)


@dataclass(frozen=True)
class FrameGroup:
    id: int
    start_date: datetime
    end_date: datetime
    hostname: str
    files: list[Path]
    metadata_by_file: dict[Path, dict]

    def merge(self, other: "FrameGroup") -> "FrameGroup":
        if self.start_date < other.start_date:
            return FrameGroup._merge(self, other)
        else:
            return FrameGroup._merge(other, self)

    @staticmethod
    def _merge(first: "FrameGroup", second: "FrameGroup") -> "FrameGroup":
        if first.hostname != second.hostname:
            raise ValueError("Hostname of FrameGroups does not match")

        files = first.files + second.files
        metadata = dict(first.metadata_by_file)
        metadata.update(second.metadata_by_file)

        merged = FrameGroup(
            id=first.id,
            start_date=first.start_date,
            end_date=second.end_date,
            hostname=first.hostname,
            files=files,
            metadata_by_file=metadata,
        )

        return merged

    def check_any_output_file_exists(self, with_suffix: str) -> bool:
        return len(self.get_existing_output_files(with_suffix)) > 0

    def get_existing_output_files(self, with_suffix: str) -> list[Path]:
        return [file for file in self.get_output_files(with_suffix) if file.is_file()]

    def get_output_files(self, with_suffix: str) -> list[Path]:
        return [get_output_file(file, with_suffix) for file in self.files]

    def with_id(self, new_id: int) -> "FrameGroup":
        return replace(self, id=new_id)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"FrameGroup[{self.id}] = [{self.start_date} - {self.end_date}]"


class FrameGroupParser(ABC):

    def process_all(self, files: list[Path]) -> list[FrameGroup]:
        files_progress = tqdm(files, desc="parse FrameGroups", total=len(files))

        parsed: list[FrameGroup] = [self.parse(file) for file in files_progress]
        merged: list[FrameGroup] = self.merge(parsed)
        updated: list[FrameGroup] = [
            self.updated_metadata_copy(group).with_id(i)
            for i, group in enumerate(merged)
        ]

        return updated

    @abstractmethod
    def parse(self, file: Path) -> FrameGroup:
        pass

    @abstractmethod
    def merge(self, frame_groups: list[FrameGroup]) -> list[FrameGroup]:
        pass

    def updated_metadata_copy(self, frame_group: FrameGroup) -> FrameGroup:
        new_metadata = self.update_metadata(frame_group)
        return replace(frame_group, metadata_by_file=new_metadata)

    @abstractmethod
    def update_metadata(self, frame_group: FrameGroup) -> dict[Path, dict]:
        pass
