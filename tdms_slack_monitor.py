#!/usr/bin/env python3
"""
TDMS File Monitor and Slack Notifier

Monitors a folder for new TDMS files, generates
diagnostic plots, and posts them to Slack.

Usage:
    python tdms_slack_monitor.py config.toml
"""

import glob
import io
import logging
import pathlib
import sys
import time
from dataclasses import dataclass, field

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tomllib
from matplotlib.ticker import MultipleLocator
from nptdms import TdmsFile
from scipy.signal import welch
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for the TDMS monitor."""

    folder_pattern: str
    slack_bot_token: str
    slack_channel_id: str
    check_interval_minutes: float = 30.0
    min_file_size_mb: float = 10.0
    segments: int = 20
    fmax: float = 200.0


@dataclass
class FileSignature:
    """Lightweight file identifier using size and mtime."""

    size: int
    mtime: float

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "FileSignature":
        stat = path.stat()
        return cls(size=stat.st_size, mtime=stat.st_mtime)


@dataclass
class ProcessedFileState:
    """Tracks which files have been processed (in-memory only)."""

    processed: dict[str, FileSignature] = field(default_factory=dict)

    def is_new_or_modified(self, filepath: str, sig: FileSignature) -> bool:
        existing = self.processed.get(filepath)
        if existing is None:
            return True
        return existing.size != sig.size or existing.mtime != sig.mtime

    def mark_processed(self, filepath: str, sig: FileSignature) -> None:
        self.processed[filepath] = sig

    def seed_from_files(self, filepaths: list[str]) -> None:
        """Mark files as already processed (used at startup)."""
        for filepath in filepaths:
            path = pathlib.Path(filepath)
            if path.exists():
                self.processed[filepath] = FileSignature.from_path(path)


def read_tdms(filename: str, groupname: str = "Group", channelname: str = "Channel"):
    """Read TDMS file and return data and properties."""
    tdms = TdmsFile.read(filename)
    group = tdms[groupname]
    channel = group[channelname]
    data = channel[:]
    props = channel.properties
    return data, props


def compute_welch_asd(
    data: np.ndarray,
    fs: float,
    segments: int,
    window: str = "hann",
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch ASD for the given data.

    Returns:
        freqs: frequency array
        asd: amplitude spectral density
    """
    nperseg = len(data) // segments
    noverlap = nperseg // 2

    freqs, psd = welch(
        data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=True,
        detrend=detrend,
    )
    asd = np.sqrt(psd)
    return freqs, asd


def create_diagnostic_plot(
    filepath: str, segments: int = 20, fmax: float = 200.0
) -> io.BytesIO:
    """
    Create a diagnostic plot for a TDMS file.

    Returns:
        BytesIO buffer containing the PNG image.
    """
    tdms_data, tdms_props = read_tdms(filepath)

    # Construct time axis
    dt = tdms_props["wf_increment"]
    fs = 1.0 / dt
    start_time = tdms_props["wf_start_time"]
    time_axis = np.arange(
        start_time,
        start_time + len(tdms_data) * np.timedelta64(int(dt * 1e9), "ns"),
        np.timedelta64(int(dt * 1e9), "ns"),
        dtype=np.datetime64,
    )

    # Compute ASD
    freqs, asd = compute_welch_asd(tdms_data, fs, segments)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Time series plot
    ax1.plot(time_axis, tdms_data, linewidth=0.5, color="cornflowerblue")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d; %H:%M"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax1.set_xlabel("UTC (YY-MM-DD; hh:mm)")
    ax1.set_ylabel(tdms_props.get("unit_string", "V"))
    ax1.set_title(pathlib.Path(filepath).name)
    ax1.grid(True, alpha=0.3)

    # ASD plot
    freq_mask = freqs <= fmax
    ax2.plot(freqs[freq_mask], asd[freq_mask], linewidth=0.8, color="cornflowerblue")
    ax2.set_xlim(0, fmax)
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"ASD ($\mathrm{V_{rms}}/\sqrt{\mathrm{Hz}}$)")
    ax2.xaxis.set_major_locator(MultipleLocator(fmax / 10))
    ax2.xaxis.set_minor_locator(MultipleLocator(fmax / 100))
    ax2.grid(True, which="both", alpha=0.2)

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)

    return buf


def post_image_to_slack(
    client: WebClient,
    channel_id: str,
    image_buffer: io.BytesIO,
    filename: str,
) -> bool:
    """Post an image to Slack using the Slack API."""
    try:
        client.files_upload_v2(
            channel=channel_id,
            file=image_buffer,
            filename=f"{filename}.png",
            title=f"Diagnostic: {filename}",
            initial_comment=f"📊 New TDMS file: `{filename}`",
        )
        logger.info(f"Posted plot to Slack: {filename}")
        return True
    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
        return False


def get_sorted_files(folder_pattern: str, min_size_bytes: int) -> list[str]:
    """Get all TDMS files sorted by modification time, filtered by size."""
    all_files = glob.glob(folder_pattern)
    valid_files = []

    for filepath in all_files:
        path = pathlib.Path(filepath)
        if path.stat().st_size >= min_size_bytes:
            valid_files.append((filepath, path.stat().st_mtime))

    # Sort by mtime (newest last)
    valid_files.sort(key=lambda x: x[1])
    return [f[0] for f in valid_files]


def find_new_files(
    folder_pattern: str,
    state: ProcessedFileState,
    min_size_bytes: int,
) -> list[str]:
    """Find new or modified TDMS files that haven't been processed."""
    all_files = get_sorted_files(folder_pattern, min_size_bytes)
    new_files = []

    for filepath in all_files:
        path = pathlib.Path(filepath)
        sig = FileSignature.from_path(path)

        if state.is_new_or_modified(filepath, sig):
            new_files.append(filepath)

    return new_files


def process_file(filepath: str, config: MonitorConfig) -> io.BytesIO | None:
    """Process a single TDMS file and return the plot buffer."""
    logger.info(f"Processing: {filepath}")

    try:
        return create_diagnostic_plot(
            filepath,
            segments=config.segments,
            fmax=config.fmax,
        )
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return None


def load_config(path: str) -> MonitorConfig:
    """Load configuration from TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return MonitorConfig(**data)


def run_monitor(config: MonitorConfig) -> None:
    """Main monitoring loop."""
    min_size_bytes = int(config.min_file_size_mb * 1e6)
    client = WebClient(token=config.slack_bot_token)
    state = ProcessedFileState()

    # Seed state with most recent 10 files to avoid reprocessing on restart
    all_files = get_sorted_files(config.folder_pattern, min_size_bytes)
    recent_files = all_files[-10:] if len(all_files) > 10 else all_files
    state.seed_from_files(recent_files)
    logger.info(f"Seeded state with {len(recent_files)} recent file(s)")

    logger.info("Starting TDMS monitor")
    logger.info(f"  Folder pattern: {config.folder_pattern}")
    logger.info(f"  Check interval: {config.check_interval_minutes} minutes")
    logger.info(f"  Min file size: {config.min_file_size_mb} MB")

    while True:
        try:
            new_files = find_new_files(
                config.folder_pattern,
                state,
                min_size_bytes,
            )

            if new_files:
                logger.info(f"Found {len(new_files)} new file(s)")

                for filepath in new_files:
                    buf = process_file(filepath, config)
                    path = pathlib.Path(filepath)
                    sig = FileSignature.from_path(path)

                    if buf is not None:
                        post_image_to_slack(client, config.slack_channel_id, buf, path.name)

                    # Mark as processed even if plotting failed (avoid retry loop)
                    state.mark_processed(filepath, sig)
            else:
                logger.info("No new files")

            logger.info(f"Sleeping for {config.check_interval_minutes} minutes...")
            time.sleep(config.check_interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(60)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.toml>", file=sys.stderr)
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_monitor(config)


if __name__ == "__main__":
    main()
