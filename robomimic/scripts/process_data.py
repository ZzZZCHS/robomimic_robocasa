from robomimic.utils.ground_utils import GroundUtils
import os
from tqdm import tqdm
import h5py
import json
import time
import numpy as np
import spacy
from PIL import Image

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Define a function to extract noun phrases
def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if not any(token.pos_ == "PRON" for token in chunk):
            noun_phrases.append(chunk.text)
    return noun_phrases


if __name__ == "__main__":
    
    breakpoint()

    grounding_model = GroundUtils(device="cuda:2")
    data_root = 'groundingLMM/robocasa_datasets'
    ori_data_dir = os.path.join(data_root, 'v0.1/single_stage')

    obs_keys = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]

    prompt_template = "Can you segment {}?"

    src_name = 'demo_gentex_im128_randcams.hdf5'
    tgt_name = 'demo_gentex_im128_randcams_addmask.hdf5'


    if not os.path.exists(ori_data_dir):
        raise FileNotFoundError

    for root, dirs, files in tqdm(os.walk(ori_data_dir)):
        for file_name in files:
            if file_name != src_name:
                continue
            if "PnPCounterToSink" not in root:
                continue
            ori_file_path = os.path.join(root, src_name)
            tgt_file_path = os.path.join(root, tgt_name)
            print(ori_file_path)
            # if os.path.exists(tgt_file_path):
            #     continue
            # shutil.copy(ori_file_path, tgt_file_path)
            # print(tgt_file_path)
            # f = h5py.File(tgt_file_path, 'a+')
            f = h5py.File(ori_file_path, 'r')
            data = f['data']
            for demo_id in tqdm(data.keys()):
                tmp_demo = data[demo_id]
                tmp_obs = tmp_demo['obs']
                ep_meta = tmp_demo.attrs.get('ep_meta')
                ep_meta = json.loads(ep_meta)
                lang = ep_meta.get('lang')
                noun_phrases = extract_noun_phrases(lang)
                
                for k in tmp_obs.keys():
                    if k in obs_keys:
                        tmp_img = tmp_obs[k][()][0]
                        # print(tmp_img.shape)
                        im = Image.fromarray(tmp_img)
                        im.save(f"{k}.jpg")
                        st_time = time.time()
                        tmp_masked_img = None
                        for i, tmp_phrase in enumerate(noun_phrases):
                            tmp_prompt = prompt_template.format(tmp_phrase)
                            print(tmp_prompt)
                            tmp_masked_img = grounding_model.inference(tmp_prompt, tmp_img, tmp_masked_img)
                            im = Image.fromarray(tmp_masked_img)
                            im.save(f"masked_img{i}.jpg")
                        tmp_masked_img = np.expand_dims(tmp_masked_img, axis=0)
                        ed_time = time.time()
                        print("time:", ed_time - st_time)
                        breakpoint()
                        # dset = tmp_obs.create_dataset(f'masked_{k}', data=tmp_masked_img)
                #         breakpoint()
                # break