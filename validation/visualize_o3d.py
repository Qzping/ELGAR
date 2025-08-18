from icecream import ic
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re
from validation import Validation
from data_process.data_manipulation import DataManipulatorCello
import ffmpeg
from concurrent.futures import ProcessPoolExecutor


def process_view(
        video_path,
        shot_path,
        i,
        npy_name,
        color_pool,
        method,
        start_f,
        end_f,
        view_select,
        view,
        audioroot,
        offset,
        selected_frame,
        if_vis,
        get_obj,
        test_split,
        sample_timestamp='2025',
        train_timestamp='2025',
        ckpt_step='90k',
        instrument='cello'
):
    if npy_name.endswith(".npy"):
        ic(npy_name)
        color_idx = i % len(color_pool)
        npy_name = npy_name.split(".npy")[0]
        color = color_pool[color_idx]
        ic(color)


        # song_name = npy_name.split('_*k_')[0]
        pattern = rf'_{ckpt_step}_'
        ic(re.split(pattern, npy_name))
        song_name = re.split(pattern, npy_name)[0]
        ic(song_name)
        audio_name = rf"{song_name}.wav"
        # audio_path = rf"../../SPD/{audioroot}/normalized"
        audio_path = rf"../dataset/{audioroot}/"
        audio_file = os.path.join(audio_path, audio_name)
        ic(audio_file)

        video_name = rf"{song_name}_{color}_{view}.mp4"
        rendered_name = rf"{song_name}_{color}_{view}_rendered.mp4"
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, video_name)
        # video_render = os.path.join(video_path, rendered_name)
        merge_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'audiovisuals')
        os.makedirs(merge_path, exist_ok=True)
        merge_file = os.path.join(merge_path, video_name)
        merge_render = os.path.join(merge_path, rendered_name)

        manual_view_name = rf"{view}.json"
        manual_view_path = rf"./manual_views/paper_views/"
        os.makedirs(manual_view_path, exist_ok=True)
        manual_view_file = os.path.join(manual_view_path, manual_view_name)

        if test_split:
            validator = Validation(generation_motion_file=rf'../save/{method}/{train_timestamp}/sample/{sample_timestamp}_{song_name}/{npy_name}.npy',  # not used by visualization
                                   original_motion_file=rf'../dataset/SPD-GEN/{instrument}/test_data/motion/{song_name}.json',
                                   original_body_file=rf'../dataset/SPD-GEN/{instrument}/test_data/motion/body_{song_name}_calc.pkl',
                                   input_audio_file=rf'../dataset/SPD-GEN/{instrument}/test_data/audio/{song_name}.wav',
                                   vis=True)
            # validator.original_cp_3d = apply_rotations_to_points(validator.original_cp_3d, validator.generation_R_s)
            validator.original_cp_3d = validator.original_cp_3d[start_f:end_f]
        else:
            validator = Validation(generation_motion_file=rf'../save/{method}/{train_timestamp}/sample/{sample_timestamp}_{song_name}/{npy_name}.npy',
                                   original_motion_file=rf'../dataset/SPD-GEN/{instrument}/test_data/motion/cello01.json',  # not used by visualization
                                   original_body_file=rf'../dataset/SPD-GEN/{instrument}/test_data/motion/body_cello01_calc.pkl',  # not used by visualization
                                   input_audio_file=rf'../dataset/SPD-GEN/{instrument}/test_data/audio/cello01.wav',  # not used by visualization
                                   vis=True)
            validator.original_cp_3d = None

        data_manipulator_cello = DataManipulatorCello(bodycolor=color)

        # Rotation
        # validator.generation_lh_3d = apply_rotations_to_points(validator.generation_lh_3d, validator.generation_R_s)
        # validator.generation_rh_3d = apply_rotations_to_points(validator.generation_rh_3d, validator.generation_R_s)
        # validator.generation_body_3d = apply_rotations_to_points(validator.generation_body_3d, validator.generation_R_s)
        # validator.generation_bow_3d = apply_rotations_to_points(validator.generation_bow_3d, validator.generation_R_s)
        # # validator.generation_instrument_3d = data_manipulator_cello.get_second_third_string(validator.generation_instrument_3d)
        # validator.generation_instrument_3d = apply_rotations_to_points(validator.generation_instrument_3d, validator.generation_R_s)

        # ic(validator.generation_lh_3d.shape)

        if view_select is True:
            data_manipulator_cello.visualize_3dkp_o3d_view_select(lefthand_3d_position=validator.generation_lh_3d[start_f:end_f],
                                                                  righthand_3d_position=validator.generation_rh_3d[start_f:end_f],
                                                                  bow_3d_position=validator.generation_bow_3d[start_f:end_f],
                                                                  body_3d_position=validator.generation_body_3d[start_f:end_f],
                                                                  instrument_3d_position=validator.generation_instrument_3d[start_f:end_f],
                                                                  cp_3d_position=validator.original_cp_3d,
                                                                  view_type="manual_view",
                                                                  manual_view_file=manual_view_file,
                                                                  if_vis=if_vis,
                                                                  selected_frame=selected_frame
                                                                  )
        else:
            print("using existing view setup")
            data_manipulator_cello.visualize_3dkp_o3d(output_path=video_file,
                                                      shot_path=shot_path,
                                                      audioname=song_name,
                                                      lefthand_3d_position=validator.generation_lh_3d[start_f:end_f],
                                                      righthand_3d_position=validator.generation_rh_3d[start_f:end_f],
                                                      bow_3d_position=validator.generation_bow_3d[start_f:end_f],
                                                      body_3d_position=validator.generation_body_3d[start_f:end_f],
                                                      instrument_3d_position=validator.generation_instrument_3d[start_f:end_f],
                                                      cp_3d_position=validator.original_cp_3d,
                                                      plot_static_frame=False,
                                                      view_type="no_type",
                                                      view_file=manual_view_file,
                                                      lookat_point=validator.generation_lh_3d[0][17],
                                                      offset=start_f,
                                                      if_vis=if_vis,
                                                      get_obj=get_obj)

        if os.path.isfile(audio_file) and view_select is False:
            audio_input = ffmpeg.input(audio_file, ss=offset)
            ffmpeg.input(video_file).output(
                audio_input,
                merge_file,
                vcodec='copy',
                acodec='aac',
                shortest=None
            ).run(quiet=True, overwrite_output=True)


def run_all_views_in_parallel(
        npy_list,
        video_path,
        shot_path,
        color_pool,
        method,
        start_f,
        end_f,
        view_select,
        view,
        audioroot,
        offset,
        selected_frame,
        if_vis,
        get_obj,
        test_split,
        sample_timestamp='2025',
        train_timestamp='2025',
        ckpt_step='90k',
        instrument='cello'
):
    with ProcessPoolExecutor() as executor:
        results = []
        for i, npy_name in enumerate(npy_list):
            results.append(executor.submit(process_view, video_path, shot_path, i, npy_name, color_pool, method,
                                           start_f, end_f, view_select, view, audioroot, offset, selected_frame,
                                           if_vis, get_obj, test_split, sample_timestamp, train_timestamp,
                                           ckpt_step, instrument))

        for result in results:
            print(result.result())

    print(f"{method} ----- tasks completed.")


if __name__ == "__main__":

    view_select = True

    methods = ['full']
    train_timestamps = ['2025']
    train_modes = ['total']
    sample_timestamps = ['2025']
    npy_list = ['vocalise_90k_g1.0_rep0.npy']
    instrument = 'cello'

    color_pool = ["blue", "orange", "purple"]

    start_f = 0
    end_f = 300
    offset = start_f / 30

    view, view_type = "front", "main"

    test_split = False  # whether the npy is generated from the test set
    selected_frame = 0
    ckpt_step = '90k'  # we use 90k

    if view_type == "bow":
        if_vis = [False, True, False, True, True, False]  # ground, instrument, lh, rh, bow, body
    elif view_type == "string":
        if_vis = [False, True, True, False, False, False]
    elif view_type == "performer":
        if_vis = [False, True, True, True, True, False]
    elif view_type == "main_noground":
        if_vis = [False, True, True, True, True, True]
    elif view_type == "instrument_only":
        if_vis = [False, True, False, False, False, False]
    elif view_type == 'bow_only':
        if_vis = [False, False, False, False, True, False]
    elif view_type == 'main':
        if_vis = [True, True, True, True, True, True]
    else:
        raise TypeError(f'View type {view_type} is not supported!')

    get_obj = False

    for i, method in enumerate(methods):
        train_mode = train_modes[i]
        train_timestamp = train_timestamps[i]
        sample_timestamp = sample_timestamps[i]
        if train_mode == "partial":
            audioroot = rf"SPD-GEN/{instrument}/test_data/audio"
        else:
            audioroot = rf"wild_data/{instrument}/audio"

        video_path = rf"o3d_results/generate_data/{view}/{method}/videos/"
        shot_path = rf"o3d_results/generate_data/{view}/{method}/shots/"
        os.makedirs(shot_path, exist_ok=True)

        run_all_views_in_parallel(npy_list, video_path, shot_path, color_pool, method, start_f, end_f,
                                  view_select, view, audioroot, offset, selected_frame, if_vis, get_obj,
                                  test_split, sample_timestamp, train_timestamp, ckpt_step, instrument)
