import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from validation import Validation
from icecream import ic
import numpy as np
import glob


def cosine_similarity(a, b):
    """
    cosine similarity computation between two vectors
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == "__main__":

    lhcd_s = []
    bcd_s = []
    bowing_precision_s = []
    bowing_recall_s = []
    bowing_f1_s = []
    cs_s = []


    models = ['hicl_0_bicl_0_partial',
              'hicl_1_bicl_0_partial',
              'hicl_1_bicl_1_partial']
    
    timestamps = ['your_timestamp_1', 'your_timestamp_2', 'your_timestamp_3']

    test_set = ['01', '05', '08', '12', '26_1', '26_2', '26_3', '45', '47', '59_1', '59_2', '60_1', '60_2', '60_3', '60_4']


    for idx, model in enumerate(models):
        print(f"Model: {model}")
        lh_cd_avg_storage = []
        bow_cd_avg_storage = []
        cs_avg_storage = []
        precision_storage = []
        recall_storage = []
        f1_storage = []

        lh_cd_testset_sum = 0
        lh_cd2_testset_sum = 0
        bow_cd_testset_sum = 0
        bow_cd_gt_sum = 0
        bow_cd_pred_sum = 0
        consine_similarity_testset_sum = 0
        test_cnt = 0    # cnt represents frames with cp
        gtsum_cnt = 0
        testset_valid_num_frame = 0     # valid_num_frame reprensents sequence length based on audio

        bowing_detection_details_sum = {
            "num_true_pred": 0,
            "num_pred": 0,
            "num_matched_gt": 0,
            "num_gt": 0
        }

        # frog-bridge_l distance
        vector_1_pred = np.array([])
        vector_1_gt = np.array([])

        # bridge -> frog vector
        vector_2_x_pred = []
        vector_2_x_gt = []
        vector_2_y_pred = []
        vector_2_y_gt = []
        vector_2_z_pred = []
        vector_2_z_gt = []


        for cello_index in test_set:
            print(f"Evalutating cello{cello_index} ...")
            
            generated_motion_file_list = glob.glob(f"../save/{model}/{timestamps[idx]}/sample/*_cello{cello_index}/*.npy")
            if len(generated_motion_file_list) != 1:
                raise Exception('Wrong number of generated motion file!')
            else:
                generated_motion_file = generated_motion_file_list[0]

            validator = Validation(generation_motion_file=generated_motion_file,
                                original_motion_file=f'../dataset/SPD-GEN/cello/test_data/motion/cello{cello_index}.json',
                               original_body_file=f"../dataset/SPD-GEN/cello/test_data/motion/body_cello{cello_index}_calc.pkl",
                                input_audio_file=f'../dataset/SPD-GEN/cello/test_data/audio/cello{cello_index}.wav')

            lh_cd2_array = validator.cal_finger_string_distance(validator.generation_lh_3d, validator.original_cp_3d, validator.used_finger_idx)
            lh_cd2_piece_sum = np.sum(lh_cd2_array[~np.isnan(lh_cd2_array)])  # non-nan sum up

            # left hand contact deviation + bow contact deviation
            lh_cd_piece_avg, bow_cd_piece_avg, cnt, distance_allframe, distance_four_tip = validator.validate_contact_deviation()
            lh_cd_piece_sum = lh_cd_piece_avg * cnt
            bow_cd_piece_sum = bow_cd_piece_avg * cnt

            bow_cd_piece_avg_gt, bow_cd_piece_sum_gt, gtpiece_cnt = validator.cal_gt_string_bow_distance(validator.original_bow_3d, validator.original_instrument_3d, validator.cp_info)
            bow_cd_piece_avg_pred, bow_cd_piece_sum_pred, predpiece_cnt = validator.cal_gt_string_bow_distance(validator.generation_bow_3d, validator.generation_instrument_3d, validator.cp_info)
            gtsum_cnt += gtpiece_cnt


            lh_cd_testset_sum += lh_cd_piece_sum
            lh_cd2_testset_sum += lh_cd2_piece_sum
            bow_cd_testset_sum += bow_cd_piece_sum
            bow_cd_gt_sum += bow_cd_piece_sum_gt
            bow_cd_pred_sum += bow_cd_piece_sum_pred
            test_cnt += cnt

            lh_cd_avg_storage.append(lh_cd_piece_avg)
            bow_cd_avg_storage.append(bow_cd_piece_avg)

            # frog-bridge_l distance
            frog_bridge_distance_pred, _, _ = validator._get_bow_changing_from_motion(validator.generation_bow_3d,
                                                                                      validator.generation_instrument_3d,
                                                                                      max_valid_frame=validator.valid_num_frame)
            frog_bridge_distance_gt, _, _ = validator._get_bow_changing_from_motion(validator.original_bow_3d,
                                                                                    validator.original_instrument_3d,
                                                                                    max_valid_frame=validator.valid_num_frame)
            vector_1_pred = np.concatenate([vector_1_pred, frog_bridge_distance_pred])
            vector_1_gt = np.concatenate([vector_1_gt, frog_bridge_distance_gt])



            # Bow metrics (F1)
            precision_piece, recall_piece, f1_piece, details_piece = validator.validate_bowing_from_audio()
            bowing_detection_details_sum["num_true_pred"] += details_piece["num_true_pred"]
            bowing_detection_details_sum["num_pred"] += details_piece["num_pred"]
            bowing_detection_details_sum["num_matched_gt"] += details_piece["num_matched_gt"]
            bowing_detection_details_sum["num_gt"] += details_piece["num_gt"]
            precision_storage.append(precision_piece)
            recall_storage.append(recall_piece)
            f1_storage.append(f1_piece)


        lh_cd_testset_avg = lh_cd_testset_sum / test_cnt
        lh_cd2_testset_avg = lh_cd2_testset_sum / gtsum_cnt

        bow_cd_testset_avg = bow_cd_testset_sum / test_cnt
        bow_cd_gt_avg = bow_cd_gt_sum / gtsum_cnt
        bow_cd_pred_avg = bow_cd_pred_sum / gtsum_cnt
        bowing_testset_precision = bowing_detection_details_sum["num_true_pred"] / bowing_detection_details_sum["num_pred"] if bowing_detection_details_sum["num_pred"] else 0
        bowing_testset_recall = bowing_detection_details_sum["num_matched_gt"] / bowing_detection_details_sum["num_gt"] if bowing_detection_details_sum["num_gt"] else 0
        bowing_testset_f1 = 2 * (bowing_testset_precision * bowing_testset_recall) / (bowing_testset_precision + bowing_testset_recall) if (bowing_testset_precision + bowing_testset_recall) > 0 else 0

        half_bow_length = 0.75 / 2  # 0.375
        vector_1_pred_minueshalfbow = vector_1_pred - half_bow_length
        vector_1_gt_minueshalfbow = vector_1_gt - half_bow_length
        cs_1_halfbow = cosine_similarity(vector_1_pred_minueshalfbow, vector_1_gt_minueshalfbow)
        # print(f"Model {model} frog-bridge distance cosine similarity: {cs_1_halfbow} (half-bow)")

        ic(lh_cd_testset_avg)
        ic(bow_cd_testset_avg)
        # ic(bowing_testset_precision)
        # ic(bowing_testset_recall)
        ic(bowing_testset_f1)

        lhcd_s.append(lh_cd_testset_avg)
        bcd_s.append(bow_cd_testset_avg)
        bowing_precision_s.append(bowing_testset_precision)
        bowing_recall_s.append(bowing_testset_recall)
        bowing_f1_s.append(bowing_testset_f1)
        cs_s.append(cs_1_halfbow)

    ic(lhcd_s)
    ic(bcd_s)
    # ic(bowing_precision_s)
    # ic(bowing_recall_s)
    ic(bowing_f1_s)
    ic(cs_s)
