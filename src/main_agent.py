import os
import sys
from pathlib import Path
        
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

import argparse
import logging
import pickle
import yaml
import subprocess
import numpy as np

from .utils.path import load_video_footages, get_data_dir
from .agent.planner_summary import get_summary
from .agent.planner import get_plan_with_llm
from .agent.director import get_segment_guidance
from .agent.editor import generate_segment_video, EditResult
from .agent.music_utils import analyze_music, get_music_info, MusicResult
from .agent.editing_utils import Score, ScoreConfig, ShotCandidate
from .agent.interaction_utils import SegmentGuidance, SectionInfo, parse_segment_guidance
from .agent.director_validation import BeamCandidate, validate_edit

logger = logging.getLogger('main')

def set_logger(log_path="output/tmp.log"):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s'))

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'))

    logger_list = ['planner', 'planner_summary', 'editor', 'director', 'director_validation', 'retriever', 'path', 'main']
    for name in logger_list:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False

def truncate_pacing(beats_remaining, PC):
    while sum(PC) > beats_remaining:
        if PC[-1] > (sum(PC) - beats_remaining):
            PC[-1] -= (sum(PC) - beats_remaining)
        else:
            PC.pop()

def main_agent() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to the task config YAML file.")
    parser.add_argument("--cfg_path", type=str, default="configs/cfg.yaml", help="Path to the system config YAML file.")
    parser.add_argument("--result_path", type=str, default="output/out.mp4", help="Path to save result video.")
    parser.add_argument("--log_path", type=str, default="output/out.log", help="Path to save log file.")
    args = parser.parse_args()

    set_logger(args.log_path)

    with open(args.yaml_path, "r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f)

    with open(args.cfg_path, "r", encoding="utf-8") as f:
        system_config = yaml.safe_load(f)

    csv_path = Path(task_config["video_csv"])
    music_path = Path(task_config["music_path"])
    user_prompt = task_config["user_prompt"]
    video_fps = task_config["video_fps"]

    # load video footages & features
    logger.info(f"Loading video footages and features from {csv_path}...")
    video_paths, video_features = load_video_footages(csv_path)

    # preprocess footage summary and music info
    footage_summary = get_summary(csv_path, video_paths, video_features, use_cache=True)

    music_prof = analyze_music(music_path)
    music_info = get_music_info(music_prof)

    # Planner: global music section plan
    if system_config["disable_planner"]:
        section_list = [
            SectionInfo(
                label=seg.label,
                energy_level="medium",
                visual_tags=[user_prompt],
                rationale="",
            )
            for seg in music_prof.segments
        ]
    else:
        section_list = get_plan_with_llm(
            user_prompt=user_prompt,
            footage_summary=footage_summary,
            music_info=music_info,
            music_section_count=len(music_prof.segments),
            retry_count=3
        )

    # music info
    lower_rms, upper_rms = music_prof.get_rms_quantiles()
    beats = music_prof.beats
    curr_beat_idx = 0
    sections = [
        (section, [beat for beat in beats if section.start <= beat < section.end])
        for section in music_prof.segments
    ]

    # Initialize
    prev_queries: list[str] = []
    result = EditResult(
        video_candidates=[],
        total_frames=0,
        total_score=0.0
    )
    # Add initial black screen segment
    result.append_shot(ShotCandidate(
        video_id="BLACK_SCREEN",
        shot_idx=-1,
        start_frame=0,
        end_frame=round(beats[0] * video_fps) - 1,  
        score=Score(0, 0, 0, 0, 0, 0)
    ))

    beam_dir = Path("tmp/beam")
    beam_dir.mkdir(parents=True, exist_ok=True)

    # Generate process per section
    for idx, curr_seg in enumerate(sections):
        section, beat_info = curr_seg
        logger.info(f"Section {section.label}: {len(beat_info)} beats")
        section_info = section_list[idx]
        beats_remaining = len(beat_info)

        while beats_remaining > 0:
            if system_config["disable_director"]:
                retrieval_query = " ".join(section_info.visual_tags)
                pacing = [4, 4, 4, 4]
                guidance = SegmentGuidance(
                    retrieval_query=retrieval_query,
                    weight_profile="Semantic_Priority",
                    pacing_control=pacing,
                )
            else:
                guidance = get_segment_guidance(
                    user_prompt=user_prompt,
                    footage_summary=footage_summary,
                    music_info=music_info,
                    section_info=section_info,
                    beats_remaining=beats_remaining,
                    prev_queries=prev_queries,
                )

            # Retry loop with validator feedback
            regen_attempts = 0
            max_attempts = system_config["max_validation_attempts"]
            while True:
                if sum(guidance.pacing_control) > beats_remaining:
                    truncate_pacing(beats_remaining, guidance.pacing_control)

                prompt_embed, pacing_control, config = parse_segment_guidance(guidance)

                temp_idx = curr_beat_idx
                cut_points: list[float] = []
                visual_energies: list[float] = []
                for len_beat in pacing_control:
                    start_time = beats[temp_idx]
                    temp_idx = min(temp_idx + len_beat, len(beats) - 1)
                    end_time = beats[temp_idx]
                    cut_points.append(beats[temp_idx])

                    shot_rms = music_prof.get_rms(start_time, end_time)
                    visual_energies.append(np.clip(100 * (shot_rms - lower_rms) / (upper_rms - lower_rms), 0, 100))

                beam_results = generate_segment_video(
                    video_features,
                    prompt_embed,
                    cut_points,
                    visual_energies,
                    config,
                    result,
                    beam_size = system_config["editor_params"]["beam_size"],
                    exploration = system_config["editor_params"]["exploration"],
                    ret_num = system_config["editor_params"]["ret_num"],
                    pool_size = system_config["editor_params"]["pool_size"],
                    fps = video_fps,
                )

                if system_config["disable_director"] or regen_attempts == max_attempts:
                    chosen = beam_results[0]
                    logger.warning(f"Validation disabled or max attempts reached. Automatically selecting the top candidate.")
                    break

                candidates: list[BeamCandidate] = []
                for j, candidate_result in enumerate(beam_results):
                    candidate_path = beam_dir / f"seg{idx}_try{regen_attempts}_cand{j}.mp4"
                    candidate_result.generate_video(video_paths, candidate_path, fps=video_fps)
                    candidates.append(
                        BeamCandidate(
                            video_path="file://" + str(candidate_path.absolute()),
                            score=candidate_result.get_score(),
                        )
                    )

                validation = validate_edit(guidance, candidates)

                if validation.is_success and validation.best_candidate is not None:
                    if validation.best_candidate is None:
                        validation.best_candidate = 0
                    chosen = beam_results[validation.best_candidate]
                    break

                regen_attempts += 1

                last_query = guidance.retrieval_query
                rejection_reason = validation.verdict or "; ".join(validation.issues) or "retrieved clips were rejected"
                guidance = get_segment_guidance(
                    user_prompt=user_prompt,
                    footage_summary=footage_summary,
                    music_info=music_info,
                    section_info=section_info,
                    beats_remaining=beats_remaining,
                    prev_queries=prev_queries,
                    last_query=last_query,
                    rejection_reason=rejection_reason,
                )

            result.extend(chosen)
            prev_queries.append(guidance.retrieval_query)
            if len(prev_queries) > 4:
                prev_queries = prev_queries[-4:]
            curr_beat_idx = temp_idx
            beats_remaining -= sum(pacing_control)

    result.generate_video(video_paths, Path("tmp.mp4"), fps=video_fps)
    # Add music
    command = [
        "ffmpeg", "-y", "-i", "tmp.mp4", "-i", str(get_data_dir() / music_path), "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", args.result_path
    ]

    subprocess.run(command, check=True)

if __name__ == "__main__":
    try:
        main_agent()
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise