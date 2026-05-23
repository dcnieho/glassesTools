from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
import json
import pathlib
import re
import statistics
from typing import Any

import av
import numpy as np
from tqdm import tqdm

from glassesTools.video_utils import get_frame_timestamps_from_video


SEGMENT_PATTERN = re.compile(r'^(?P<prefix>.+)_R(?P<index>\d+)$')
PREFERRED_ENCODERS = {
    'h264': 'libx264',
    'hevc': 'libx265',
}


@dataclass(slots=True)
class SegmentPlan:
    prefix: str
    recording_index: int
    video_path: pathlib.Path
    info_path: pathlib.Path
    raw_frame_count: int
    absolute_timestamps: list[Fraction]
    dropped_leading_frames: int = 0


@dataclass(slots=True)
class VideoSpec:
    codec_name: str
    width: int
    height: int
    pix_fmt: str | None
    time_base: Fraction
    frame_rate: Fraction | None
    bit_rate: int | None
    gop_size: int | None


@dataclass(slots=True)
class SegmentEncodePlan:
    segment: SegmentPlan
    gap_timestamps: list[Fraction]
    segment_timestamps: list[Fraction]


@dataclass(slots=True)
class GroupPlan:
    prefix: str
    source_spec: VideoSpec
    encode_plan: list[SegmentEncodePlan]
    expected_timestamps: list[Fraction]
    durations: list[Fraction]
    output_path: pathlib.Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Concatenate FLEX recording segments using paired _info.json frame timestamps.',
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        type=pathlib.Path,
        help='One or more FLEX segment files and/or directories that contain files like cam1_R001.mp4.',
    )
    parser.add_argument(
        '--output-dir',
        type=pathlib.Path,
        help='Directory for the concatenated output. Defaults to the directory of the first segment in each group.',
    )
    parser.add_argument(
        '--timestamp-field',
        default='systemTimeStamp',
        help='Field to read from each _info.json entry. Defaults to systemTimeStamp.',
    )
    parser.add_argument(
        '--codec',
        help='Override the output encoder. Defaults to the closest available encoder for the first source file.',
    )
    parser.add_argument(
        '--crf',
        type=int,
        help='Optional constant rate factor for encoders that support it.',
    )
    parser.add_argument(
        '--preset',
        help='Optional encoder preset, for example medium or slow.',
    )
    parser.add_argument(
        '--bitrate',
        help='Optional output bitrate, for example 20M or 4500k.',
    )
    parser.add_argument(
        '--pixel-format',
        help='Optional output pixel format. Defaults to the first source stream pixel format when supported.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite an existing output file.',
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify an existing concatenated output; do not re-encode.',
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip post-encode timestamp verification.',
    )
    parser.add_argument(
        '--verify-tolerance-ms',
        type=float,
        help='Allowed absolute timestamp error during verification, in milliseconds. Defaults to one output time-base tick.',
    )
    args = parser.parse_args()
    if args.verify_only and args.skip_verify:
        parser.error('--verify-only and --skip-verify cannot be used together.')
    return args


def parse_segment_stem(path: pathlib.Path) -> tuple[str, int]:
    match = SEGMENT_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f'Expected a segment name like cam1_R001.mp4, got {path.name!r}')
    return match.group('prefix'), int(match.group('index'))


def discover_groups(inputs: list[pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    groups: dict[str, list[pathlib.Path]] = {}
    seen: set[pathlib.Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            candidates = sorted(p for p in path.glob('*.mp4') if SEGMENT_PATTERN.match(p.stem))
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f'Input path does not exist: {path}')

        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            prefix, recording_index = parse_segment_stem(candidate)
            groups.setdefault(prefix, []).append(candidate)

    if not groups:
        raise FileNotFoundError('No FLEX segment files matching *_R###.mp4 were found.')

    for prefix, video_paths in groups.items():
        video_paths.sort(key=lambda path: parse_segment_stem(path)[1])
        expected = list(range(parse_segment_stem(video_paths[0])[1], parse_segment_stem(video_paths[0])[1] + len(video_paths)))
        got = [parse_segment_stem(path)[1] for path in video_paths]
        if got != expected:
            print(f'Warning: {prefix} segments are not consecutive: {got}')

    return dict(sorted(groups.items()))


def load_segment_plan(video_path: pathlib.Path, timestamp_field: str) -> SegmentPlan:
    prefix, recording_index = parse_segment_stem(video_path)
    info_path = video_path.with_name(f'{video_path.stem}_info.json')
    if not info_path.is_file():
        raise FileNotFoundError(f'Metadata file not found for {video_path.name}: {info_path.name}')

    rows = json.loads(info_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{info_path} does not contain a non-empty JSON array.')
    if timestamp_field not in rows[0]:
        raise KeyError(f'{timestamp_field!r} was not found in {info_path.name}. Available keys include: {sorted(rows[0].keys())}')

    timestamps = [Fraction(str(row[timestamp_field])) for row in rows]
    if any(curr <= prev for prev, curr in pairwise(timestamps)):
        raise ValueError(f'Timestamps in {info_path.name} are not strictly increasing.')

    return SegmentPlan(
        prefix=prefix,
        recording_index=recording_index,
        video_path=video_path,
        info_path=info_path,
        raw_frame_count=len(timestamps),
        absolute_timestamps=timestamps,
    )


def inspect_video_spec(video_path: pathlib.Path) -> VideoSpec:
    spec: VideoSpec | None = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        time_base = stream.time_base or codec_context.time_base
        if time_base is None:
            raise RuntimeError(f'Could not determine a video time base for {video_path}')

        frame_rate = None
        for candidate in (stream.average_rate, stream.base_rate, stream.guessed_rate):
            if candidate is not None:
                frame_rate = Fraction(candidate.numerator, candidate.denominator)
                break

        try:
            gop_size = codec_context.gop_size or None
        except RuntimeError:
            gop_size = None

        spec = VideoSpec(
            codec_name=codec_context.codec.name,
            width=codec_context.width,
            height=codec_context.height,
            pix_fmt=codec_context.pix_fmt,
            time_base=Fraction(time_base.numerator, time_base.denominator),
            frame_rate=frame_rate,
            bit_rate=codec_context.bit_rate or None,
            gop_size=gop_size,
        )
    if spec is None:
        raise RuntimeError(f'Could not inspect video stream metadata for {video_path}')
    return spec


def validate_compatible_specs(video_paths: list[pathlib.Path], reference_spec: VideoSpec) -> None:
    for video_path in video_paths[1:]:
        spec = inspect_video_spec(video_path)
        if (spec.width, spec.height) != (reference_spec.width, reference_spec.height):
            raise ValueError(
                f'All segments must have the same dimensions. {video_path.name} is {spec.width}x{spec.height}, '
                f'expected {reference_spec.width}x{reference_spec.height}.'
            )
        if spec.pix_fmt != reference_spec.pix_fmt:
            print(f'Warning: {video_path.name} uses pixel format {spec.pix_fmt}, expected {reference_spec.pix_fmt}.')
        if spec.codec_name != reference_spec.codec_name:
            print(f'Warning: {video_path.name} uses codec {spec.codec_name}, expected {reference_spec.codec_name}.')
        if spec.time_base != reference_spec.time_base:
            print(f'Warning: {video_path.name} uses time base {spec.time_base}, expected {reference_spec.time_base}.')


def trim_segment_overlaps(segments: list[SegmentPlan]) -> None:
    previous_end: Fraction | None = None
    for segment in segments:
        if previous_end is not None:
            while segment.absolute_timestamps and segment.absolute_timestamps[0] <= previous_end:
                segment.absolute_timestamps.pop(0)
                segment.dropped_leading_frames += 1
            if not segment.absolute_timestamps:
                raise ValueError(f'All frames in {segment.video_path.name} overlap with the previous segment.')
        previous_end = segment.absolute_timestamps[-1]


def infer_nominal_ifi(segments: list[SegmentPlan], fallback_rate: Fraction | None) -> Fraction:
    diffs = [curr - prev for segment in segments for prev, curr in pairwise(segment.absolute_timestamps)]
    if diffs:
        return statistics.median(diffs)
    if fallback_rate is not None:
        return Fraction(fallback_rate.denominator, fallback_rate.numerator)
    raise RuntimeError('Could not infer a frame interval from the segment metadata.')


def interpolate_gap(previous_ts: Fraction, next_ts: Fraction, nominal_ifi: Fraction) -> list[Fraction]:
    gap = next_ts - previous_ts
    if gap <= nominal_ifi * Fraction(3, 2):
        return []

    interval_count = int(round(float(gap / nominal_ifi)))
    missing_frames = max(interval_count - 1, 0)
    if missing_frames == 0:
        return []

    step = gap / (missing_frames + 1)
    return [previous_ts + step * idx for idx in range(1, missing_frames + 1)]


def make_encode_plan(segments: list[SegmentPlan], nominal_ifi: Fraction) -> tuple[list[SegmentEncodePlan], list[Fraction]]:
    start_time = segments[0].absolute_timestamps[0]
    previous_end: Fraction | None = None
    encode_plan: list[SegmentEncodePlan] = []
    expected_timestamps: list[Fraction] = []

    for segment in segments:
        relative_timestamps = [timestamp - start_time for timestamp in segment.absolute_timestamps]
        gap_timestamps = [] if previous_end is None else interpolate_gap(previous_end, relative_timestamps[0], nominal_ifi)
        encode_plan.append(
            SegmentEncodePlan(
                segment=segment,
                gap_timestamps=gap_timestamps,
                segment_timestamps=relative_timestamps,
            )
        )
        expected_timestamps.extend(gap_timestamps)
        expected_timestamps.extend(relative_timestamps)
        previous_end = relative_timestamps[-1]

    return encode_plan, expected_timestamps


def compute_durations(timestamps: list[Fraction], default_duration: Fraction) -> list[Fraction]:
    if not timestamps:
        raise ValueError('At least one timestamp is required to compute frame durations.')
    durations = [curr - prev for prev, curr in pairwise(timestamps)]
    durations.append(default_duration)
    if any(duration <= 0 for duration in durations):
        raise ValueError('Frame durations must be strictly positive.')
    return durations


def round_fraction(value: Fraction) -> int:
    return int(value + Fraction(1, 2))


def seconds_to_ticks(seconds: Fraction, time_base: Fraction) -> int:
    return round_fraction(seconds / time_base)


def quantized_timestamps_ms(timestamps: list[Fraction], time_base: Fraction) -> np.ndarray:
    return np.array(
        [float(seconds_to_ticks(timestamp, time_base) * time_base * 1000) for timestamp in timestamps],
        dtype=float,
    )


def parse_bitrate(value: str) -> int:
    match = re.fullmatch(r'(?i)(\d+(?:\.\d+)?)([kmg])?', value.strip())
    if not match:
        raise ValueError(f'Could not parse bitrate {value!r}. Use values like 20M, 4500k, or 2500000.')
    number = float(match.group(1))
    suffix = (match.group(2) or '').lower()
    scale = {'': 1, 'k': 1_000, 'm': 1_000_000, 'g': 1_000_000_000}[suffix]
    return int(number * scale)


def select_encoder(source_codec_name: str, override: str | None) -> str:
    if override:
        return override
    return PREFERRED_ENCODERS.get(source_codec_name, source_codec_name)


def build_encoder_options(args: argparse.Namespace) -> dict[str, str]:
    options: dict[str, str] = {}
    if args.crf is not None:
        options['crf'] = str(args.crf)
    if args.preset:
        options['preset'] = args.preset
    return options


def configure_output_stream(
    container: Any,
    source_spec: VideoSpec,
    args: argparse.Namespace,
) -> Any:
    codec_name = select_encoder(source_spec.codec_name, args.codec)
    options = build_encoder_options(args)
    output_stream = container.add_stream(codec_name, rate=source_spec.frame_rate, options=options or None)
    output_stream.width = source_spec.width
    output_stream.height = source_spec.height
    desired_pix_fmt = args.pixel_format or source_spec.pix_fmt
    if desired_pix_fmt:
        try:
            output_stream.pix_fmt = desired_pix_fmt
        except ValueError:
            print(f'Warning: encoder {codec_name} does not accept pixel format {desired_pix_fmt}; using encoder default.')
    output_stream.time_base = source_spec.time_base
    output_stream.codec_context.time_base = source_spec.time_base
    if args.bitrate:
        output_stream.bit_rate = parse_bitrate(args.bitrate)
    elif source_spec.bit_rate:
        output_stream.bit_rate = source_spec.bit_rate
    if source_spec.gop_size:
        output_stream.gop_size = source_spec.gop_size
    return output_stream


def ensure_no_remaining_frames(decoder, video_path: pathlib.Path) -> None:
    extra_frame = next(decoder, None)
    if extra_frame is not None:
        raise RuntimeError(f'{video_path.name} contains more decodable frames than its metadata file.')


def open_segment_decoder(segment: SegmentPlan):
    container = av.open(str(segment.video_path))
    stream = container.streams.video[0]
    decoder = container.decode(stream)
    try:
        for _ in range(segment.dropped_leading_frames):
            next(decoder)
    except StopIteration as exc:
        container.close()
        raise RuntimeError(
            f'{segment.video_path.name} contains fewer frames than required by the overlap-trimmed metadata.'
        ) from exc
    return container, decoder


def encode_frame(
    frame: Any,
    output_stream: Any,
    output_container: Any,
    timestamp: Fraction,
    duration: Fraction,
) -> None:
    frame.pts = seconds_to_ticks(timestamp, output_stream.time_base)
    frame.duration = max(1, seconds_to_ticks(duration, output_stream.time_base))
    frame.time_base = output_stream.time_base
    for packet in output_stream.encode(frame):
        output_container.mux(packet)


def encode_concat(
    prefix: str,
    encode_plan: list[SegmentEncodePlan],
    expected_timestamps: list[Fraction],
    durations: list[Fraction],
    source_spec: VideoSpec,
    output_path: pathlib.Path,
    args: argparse.Namespace,
) -> None:
    black_template = np.zeros((source_spec.height, source_spec.width, 3), dtype=np.uint8)
    cursor = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_desc = f'[{prefix}] writing frames'

    with av.open(str(output_path), 'w', format='mp4') as output_container:
        output_stream = configure_output_stream(output_container, source_spec, args)
        with tqdm(total=len(expected_timestamps), desc=progress_desc, unit='frame', dynamic_ncols=True) as progress_bar:
            for plan in encode_plan:
                for _ in plan.gap_timestamps:
                    frame = av.VideoFrame.from_ndarray(black_template, format='rgb24')
                    encode_frame(frame, output_stream, output_container, expected_timestamps[cursor], durations[cursor])
                    cursor += 1
                    progress_bar.update(1)

                input_container, decoder = open_segment_decoder(plan.segment)
                try:
                    for _ in plan.segment_timestamps:
                        try:
                            frame = next(decoder)
                        except StopIteration as exc:
                            raise RuntimeError(
                                f'{plan.segment.video_path.name} contains fewer decodable frames than {plan.segment.info_path.name}.'
                            ) from exc
                        encode_frame(frame, output_stream, output_container, expected_timestamps[cursor], durations[cursor])
                        cursor += 1
                        progress_bar.update(1)
                    ensure_no_remaining_frames(decoder, plan.segment.video_path)
                finally:
                    input_container.close()

        for packet in output_stream.encode(None):
            output_container.mux(packet)

    if cursor != len(expected_timestamps):
        raise RuntimeError('Internal error: not all planned timestamps were encoded.')


def verify_output(
    output_path: pathlib.Path,
    expected_timestamps: list[Fraction],
    time_base: Fraction,
    tolerance_ms: float | None,
) -> tuple[int, float]:
    actual_df = get_frame_timestamps_from_video(output_path)
    actual_ms = actual_df['timestamp'].to_numpy(dtype=float)
    expected_ms = quantized_timestamps_ms(expected_timestamps, time_base)
    if actual_ms.shape[0] != expected_ms.shape[0]:
        raise RuntimeError(
            f'Verification failed for {output_path.name}: expected {expected_ms.shape[0]} frames, got {actual_ms.shape[0]}.'
        )

    errors = np.abs(actual_ms - expected_ms)
    max_error_ms = float(errors.max()) if errors.size else 0.0
    allowed_error_ms = tolerance_ms if tolerance_ms is not None else float(time_base * 1000)
    if max_error_ms > allowed_error_ms + 1e-9:
        raise RuntimeError(
            f'Verification failed for {output_path.name}: maximum timestamp error {max_error_ms:.6f} ms exceeds '
            f'tolerance {allowed_error_ms:.6f} ms.'
        )
    return actual_ms.shape[0], max_error_ms


def resolve_output_path(prefix: str, video_paths: list[pathlib.Path], args: argparse.Namespace) -> pathlib.Path:
    output_dir = args.output_dir.resolve() if args.output_dir else video_paths[0].parent
    return output_dir / f'{prefix}_concat.mp4'


def prepare_group(prefix: str, video_paths: list[pathlib.Path], args: argparse.Namespace) -> GroupPlan:
    segments = [load_segment_plan(video_path, args.timestamp_field) for video_path in video_paths]
    trim_segment_overlaps(segments)

    source_spec = inspect_video_spec(video_paths[0])
    validate_compatible_specs(video_paths, source_spec)

    nominal_ifi = infer_nominal_ifi(segments, source_spec.frame_rate)
    encode_plan, expected_timestamps = make_encode_plan(segments, nominal_ifi)
    durations = compute_durations(expected_timestamps, nominal_ifi)

    return GroupPlan(
        prefix=prefix,
        source_spec=source_spec,
        encode_plan=encode_plan,
        expected_timestamps=expected_timestamps,
        durations=durations,
        output_path=resolve_output_path(prefix, video_paths, args),
    )


def print_group_summary(group_plan: GroupPlan) -> None:
    gap_frame_count = sum(len(plan.gap_timestamps) for plan in group_plan.encode_plan)
    segment_frame_count = sum(len(plan.segment_timestamps) for plan in group_plan.encode_plan)
    source_spec = group_plan.source_spec
    print(f'[{group_plan.prefix}] source codec={source_spec.codec_name} size={source_spec.width}x{source_spec.height} tb={source_spec.time_base}')
    print(f'[{group_plan.prefix}] segments={len(group_plan.encode_plan)} frames={segment_frame_count} inserted_black_frames={gap_frame_count}')
    for plan in group_plan.encode_plan[1:]:
        if plan.gap_timestamps:
            gap_seconds = float(plan.segment_timestamps[0] - plan.gap_timestamps[0]) if len(plan.gap_timestamps) == 1 else float(plan.gap_timestamps[-1] - plan.gap_timestamps[0])
            print(
                f'[{group_plan.prefix}] gap before {plan.segment.video_path.name}: {len(plan.gap_timestamps)} black frames '
                f'(span {gap_seconds:.6f} s between synthetic frames)'
            )


def verify_group(group_plan: GroupPlan, tolerance_ms: float | None) -> tuple[int, float]:
    verified_frames, max_error_ms = verify_output(
        group_plan.output_path,
        group_plan.expected_timestamps,
        group_plan.source_spec.time_base,
        tolerance_ms,
    )
    print(f'[{group_plan.prefix}] verified {verified_frames} frames, max timestamp error {max_error_ms:.6f} ms')
    return verified_frames, max_error_ms


def concatenate_group(prefix: str, video_paths: list[pathlib.Path], args: argparse.Namespace) -> pathlib.Path:
    group_plan = prepare_group(prefix, video_paths, args)
    print_group_summary(group_plan)

    if args.verify_only:
        if not group_plan.output_path.is_file():
            raise FileNotFoundError(f'Output file not found for verification: {group_plan.output_path}')
        print(f'[{prefix}] verifying existing output {group_plan.output_path}')
        verify_group(group_plan, args.verify_tolerance_ms)
        return group_plan.output_path

    if group_plan.output_path.exists() and not args.force:
        raise FileExistsError(f'Output file already exists: {group_plan.output_path}. Use --force to overwrite it.')

    encode_concat(
        prefix,
        group_plan.encode_plan,
        group_plan.expected_timestamps,
        group_plan.durations,
        group_plan.source_spec,
        group_plan.output_path,
        args,
    )

    if args.skip_verify:
        print(f'[{prefix}] wrote {group_plan.output_path} (verification skipped)')
        return group_plan.output_path

    verify_group(group_plan, args.verify_tolerance_ms)
    print(f'[{prefix}] wrote {group_plan.output_path}')
    return group_plan.output_path


def main() -> int:
    args = parse_args()
    groups = discover_groups(args.inputs)
    for prefix, video_paths in groups.items():
        concatenate_group(prefix, video_paths, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())