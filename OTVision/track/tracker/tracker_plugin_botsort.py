"""
BoT-SORT tracker implementation for OTVision.

Based on the BoT-SORT: Robust Associations Multi-Pedestrian Tracking paper
and implementation from https://github.com/NirAharon/BoT-SORT
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
from pathlib import Path

from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import Detection, TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker
from OTVision.track.tracker.reid_utils import SimpleReIDExtractor, DeepReIDExtractor, cosine_similarity


@dataclass
class BotSortTrack:
    """Represents a single track in BoT-SORT tracker."""
    
    def __init__(self, track_id: TrackId, detection: Detection, frame_no: FrameNo):
        self.track_id = track_id
        self.detections = [detection]
        self.frame_nos = [frame_no]
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.state = 'Tentative'  # Tentative, Confirmed, Deleted
        self.features = None  # Store re-ID features
        self.smooth_feat = None  # Smoothed features for re-ID
        
        # Initialize Kalman filter state
        # Convert Detection (x, y, w, h) to (xmin, ymin, xmax, ymax)
        xmin = detection.x - detection.w / 2
        ymin = detection.y - detection.h / 2
        xmax = detection.x + detection.w / 2
        ymax = detection.y + detection.h / 2
        
        self.mean = np.array([
            detection.x,  # center_x
            detection.y,  # center_y
            detection.w,  # width
            detection.h,  # height
            0, 0, 0, 0   # velocities
        ], dtype=np.float32)
        
        self.covariance = np.eye(8, dtype=np.float32)
        self.covariance[4:, 4:] *= 1000  # High uncertainty for velocities
        
    def predict(self) -> None:
        """Predict the next state using Kalman filter."""
        # Simple constant velocity model
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = 1  # x += vx
        F[1, 5] = 1  # y += vy
        F[2, 6] = 1  # w += vw
        F[3, 7] = 1  # h += vh
        
        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T
        
        # Add process noise
        Q = np.eye(8, dtype=np.float32)
        Q[4:, 4:] *= 0.01  # Process noise for velocities
        self.covariance += Q
        
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection: Detection, frame_no: FrameNo) -> None:
        """Update track with new detection."""
        measurement = np.array([
            detection.x,  # center_x
            detection.y,  # center_y
            detection.w,  # width
            detection.h   # height
        ], dtype=np.float32)
        
        # Kalman filter update
        H = np.eye(4, 8, dtype=np.float32)  # Measurement matrix
        R = np.eye(4, dtype=np.float32) * 0.1  # Measurement noise
        
        y = measurement - H @ self.mean  # Innovation
        S = H @ self.covariance @ H.T + R  # Innovation covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.mean = self.mean + K @ y
        self.covariance = self.covariance - K @ H @ self.covariance
        
        self.detections.append(detection)
        self.frame_nos.append(frame_no)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'
    
    def get_predicted_bbox(self) -> Tuple[float, float, float, float]:
        """Get predicted bounding box in (xmin, ymin, xmax, ymax) format."""
        cx, cy, w, h = self.mean[:4]
        return (
            cx - w/2,  # xmin
            cy - h/2,  # ymin
            cx + w/2,  # xmax
            cy + h/2   # ymax
        )
    
    def get_iou_with_detection(self, detection: Detection) -> float:
        """Calculate IoU between predicted bbox and detection."""
        pred_bbox = self.get_predicted_bbox()
        # Convert Detection (x, y, w, h) to (xmin, ymin, xmax, ymax)
        det_bbox = (
            detection.x - detection.w / 2,  # xmin
            detection.y - detection.h / 2,  # ymin
            detection.x + detection.w / 2,  # xmax
            detection.y + detection.h / 2   # ymax
        )
        
        # Calculate intersection
        x1 = max(pred_bbox[0], det_bbox[0])
        y1 = max(pred_bbox[1], det_bbox[1])
        x2 = min(pred_bbox[2], det_bbox[2])
        y2 = min(pred_bbox[3], det_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        area2 = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class BotSortTracker(Tracker):
    """BoT-SORT tracker implementation."""
    
    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track
    
    def __init__(self, get_current_config: GetCurrentConfig):
        super().__init__()
        self._get_current_config = get_current_config
        self.tracks: List[BotSortTrack] = []
        self.frame_count = 0
        
        # Initialize re-ID feature extractor
        reid_model_path = getattr(self.config.botsort, 'reid_model_path', '')
        if reid_model_path and Path(reid_model_path).exists():
            self.reid_extractor = DeepReIDExtractor(reid_model_path)
        else:
            self.reid_extractor = SimpleReIDExtractor()
        
    @property
    def track_high_thresh(self) -> float:
        """High confidence threshold for track initialization."""
        return getattr(self.config.botsort, 'track_high_thresh', 0.6)
    
    @property
    def track_low_thresh(self) -> float:
        """Low confidence threshold for track continuation."""
        return getattr(self.config.botsort, 'track_low_thresh', 0.1)
    
    @property
    def new_track_thresh(self) -> float:
        """Threshold for creating new tracks."""
        return getattr(self.config.botsort, 'new_track_thresh', 0.7)
    
    @property
    def track_buffer(self) -> int:
        """Number of frames to keep tracks without updates."""
        return getattr(self.config.botsort, 'track_buffer', 30)
    
    @property
    def match_thresh(self) -> float:
        """IoU threshold for matching detections to tracks."""
        return getattr(self.config.botsort, 'match_thresh', 0.8)
    
    def _extract_features(self, detection: Detection, image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract re-ID features from detection."""
        if image is None:
            return None
        
        # Convert detection to bbox format
        bbox = (
            detection.x - detection.w / 2,  # xmin
            detection.y - detection.h / 2,  # ymin
            detection.x + detection.w / 2,  # xmax
            detection.y + detection.h / 2   # ymax
        )
        
        # Extract features using the re-ID model
        features = self.reid_extractor.extract_features(image, bbox)
        return features
    
    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:
        """Track objects in a single frame."""
        self.frame_count += 1
        
        # Filter detections by confidence
        high_conf_detections = [d for d in frame.detections if d.conf >= self.track_high_thresh]
        low_conf_detections = [d for d in frame.detections if self.track_low_thresh <= d.conf < self.track_high_thresh]
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # First association: high confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == 'Confirmed']
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate(
            high_conf_detections, confirmed_tracks
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            confirmed_tracks[track_idx].update(high_conf_detections[det_idx], frame.no)
        
        # Second association: remaining high conf detections with unconfirmed tracks
        unconfirmed_tracks = [t for t in self.tracks if t.state == 'Tentative']
        remaining_detections = [high_conf_detections[i] for i in unmatched_detections]
        
        matched_tracks_2, unmatched_detections_2, unmatched_tracks_2 = self._associate(
            remaining_detections, unconfirmed_tracks
        )
        
        # Update matched unconfirmed tracks
        for track_idx, det_idx in matched_tracks_2:
            unconfirmed_tracks[track_idx].update(remaining_detections[det_idx], frame.no)
        
        # Third association: low confidence detections with unmatched tracks
        all_unmatched_tracks = [confirmed_tracks[i] for i in unmatched_tracks] + \
                              [unconfirmed_tracks[i] for i in unmatched_tracks_2]
        
        matched_tracks_3, _, _ = self._associate(low_conf_detections, all_unmatched_tracks)
        
        # Update tracks matched with low confidence detections
        for track_idx, det_idx in matched_tracks_3:
            all_unmatched_tracks[track_idx].update(low_conf_detections[det_idx], frame.no)
        
        # Create new tracks for unmatched high confidence detections
        final_unmatched_detections = [remaining_detections[i] for i in unmatched_detections_2]
        for detection in final_unmatched_detections:
            if detection.conf >= self.new_track_thresh:
                track_id = next(id_generator)
                new_track = BotSortTrack(track_id, detection, frame.no)
                self.tracks.append(new_track)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        
        # Mark tracks for deletion/finishing
        finished_track_ids = []
        discarded_track_ids = []
        
        for track in self.tracks:
            if track.time_since_update > self.track_buffer:
                if track.state == 'Confirmed' and track.hits >= 5:
                    finished_track_ids.append(track.track_id)
                else:
                    discarded_track_ids.append(track.track_id)
        
        # Create tracked detections
        tracked_detections = []
        for track in self.tracks:
            if track.time_since_update == 0:  # Track was updated this frame
                latest_detection = track.detections[-1]
                is_new = track.hits == 1
                tracked_detections.append(latest_detection.of_track(track.track_id, is_new))
        
        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=tracked_detections,
            image=frame.image,
            finished_tracks=set(finished_track_ids),
            discarded_tracks=set(discarded_track_ids),
        )
    
    def _associate(
        self, 
        detections: List[Detection], 
        tracks: List[BotSortTrack]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using IoU matching."""
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = track.get_iou_with_detection(detection)
        
        # Simple greedy matching (could be improved with Hungarian algorithm)
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Find matches above threshold
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
                break
                
            # Find best match
            best_iou = 0
            best_track_idx = -1
            best_det_idx = -1
            
            for t_idx in unmatched_tracks:
                for d_idx in unmatched_detections:
                    if iou_matrix[t_idx, d_idx] > best_iou and iou_matrix[t_idx, d_idx] >= self.match_thresh:
                        best_iou = iou_matrix[t_idx, d_idx]
                        best_track_idx = t_idx
                        best_det_idx = d_idx
            
            if best_track_idx == -1:
                break
                
            matched_tracks.append((best_track_idx, best_det_idx))
            unmatched_tracks.remove(best_track_idx)
            unmatched_detections.remove(best_det_idx)
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _associate_with_reid(
        self, detections: List[Detection], tracks: List[BotSortTrack], thresh: float = 0.7
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using re-ID features."""
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Extract features for detections if not already done
        det_features = []
        for det in detections:
            if hasattr(det, 'features') and det.features is not None:
                det_features.append(det.features)
            else:
                det_features.append(None)
        
        # Get track features
        track_features = []
        for track in tracks:
            if track.smooth_feat is not None:
                track_features.append(track.smooth_feat)
            else:
                track_features.append(None)
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(detections), len(tracks)))
        for i, det_feat in enumerate(det_features):
            for j, track_feat in enumerate(track_features):
                if det_feat is not None and track_feat is not None:
                    similarity_matrix[i, j] = cosine_similarity(det_feat, track_feat)
        
        # Find matches using Hungarian algorithm (simplified greedy approach)
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Greedy matching based on similarity
        while True:
            max_sim = 0
            max_i, max_j = -1, -1
            
            for i in unmatched_detections:
                for j in unmatched_tracks:
                    if similarity_matrix[i, j] > max_sim and similarity_matrix[i, j] > thresh:
                        max_sim = similarity_matrix[i, j]
                        max_i, max_j = i, j
            
            if max_i == -1:  # No more matches above threshold
                break
                
            matched_tracks.append((max_i, max_j))
            unmatched_detections.remove(max_i)
            unmatched_tracks.remove(max_j)
        
        return matched_tracks, unmatched_detections, unmatched_tracks