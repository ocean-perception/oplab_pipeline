from datetime import datetime, timezone
from pathlib import Path

from auv_nav.sensors import Camera
from oplab import Console


class VoyisLaser:
    """Load timestamped Voyis laser detections for navigation processing."""

    def __init__(self,
                 detection_path : Path) -> None:

        self.detection_path = Path(detection_path)


    @staticmethod
    def _timestamp(path : Path) -> float:
        """Convert a timestamped Voyis .pix filename to Unix epoch time."""

        timestamp = datetime.strptime(path.stem, 
                                      "%Y%m%d_%H%M%S_%f").replace(tzinfo = timezone.utc)

        return timestamp.timestamp()


    def observations(self,
                     start_time : float,
                     end_time   : float) -> list[Camera]:
        """Load Voyis laser detections within the requested time interval."""
        
        observations = []

        if not self.detection_path.is_dir():
            Console.warn(f"Voyis laser detection folder not found: "
                         f"{self.detection_path}")
            
            return observations

        for path in sorted(self.detection_path.rglob("*.pix")):

            try:
                timestamp = self._timestamp(path)

            except ValueError:
                Console.warn(f"Skipping unexpected Voyis laser filename: {path.name}")
                continue

            if not start_time <= timestamp <= end_time:
                continue

            relative_path = path.relative_to(self.detection_path)

            observation                 = Camera()
            observation.epoch_timestamp = timestamp
            observation.filename        = str((Path("image") / relative_path).with_suffix(".tif"))

            observations.append(observation)

        Console.info(f"Found {len(observations)} Voyis laser observations")

        return observations