import json
import multiprocessing as mp
import os
import sys
import pickle
"""
This script used to :
[1]. The key of 'feat' will be replaced by a real 'afeat' and shape by ashape
[2]. Add 'afeat_wav'
[3]. Add 'vfeat' key which a numpy format file.
[4]. Add 'vshape'
"""

def processing(i, oridata, pkldata, vdata, wavscp_dct, dataset):
    # [1] Replace feat by real afeat, shape by ashape (here: feat is not the input data, just a template)
    oridata["utts"][i]["input"][0]["afeat"] = oridata["utts"][i]["input"][0].pop("feat")
    oridata["utts"][i]["input"][0]["ashape"] = oridata["utts"][i]["input"][0].pop("shape")
    
    # [2] Add afeat_wav
    oridata["utts"][i]["input"][0]["afeat_wav"] = wavscp_dct[i]
    
    # [3] Add vfeat and vshape
    # original lipemb_path's test val and replay are cut by 5.8 s, 
    # so we extracted new lip embed under the 6.0 s since test val and replay duration are lower than 6.0 s
    lip_path = pkldata[i]["lipemb_path"][0]  
    # if dataset in ['Test','Replay']:
    #     lip_path = lip_path.replace('test', 'test_new')
    # if dataset in ['Val']:
    #     lip_path = lip_path.replace('val', 'val_new')
    oridata["utts"][i]["input"][0]["vfeat"] = lip_path
    # import pdb; pdb.set_trace()
    v_i = i.split("-", 1)[-1]
    if dataset == 'Replay':
        v_i = i
    oridata["utts"][i]["input"][0]["vshape"] = vdata["utts"][v_i]["input"][0]["shape"]
        
    return {i: oridata["utts"][i]}


def product_helper(args):
    return processing(*args)


def redump(
    dumpfile,
    oridumpfile,
    pickle_file,
    vdumpfile,
    wavscp,
    dataset,
    ifmulticore
):
    if ifmulticore == "true":
        ifmulticore = True
    else:
        ifmulticore = False

    # find json file and load it
    output = {"utts": {}}
    # audio
    for root, dirs, files in os.walk(oridumpfile):
        # import pdb; pdb.set_trace()
        for file in files:
            if ".json" in file:
                jsonname = file
                filename = os.path.join(root, file)
    with open(filename, encoding="UTF-8") as json_file:
        oridata = json.load(json_file)
    # video
    for root, dirs, files in os.walk(vdumpfile):
        for file in files:
            if ".json" in file:
                vjsonname = file
                vfilename = os.path.join(root, file)
    with open(vfilename, encoding="UTF-8") as json_file:
        vdata = json.load(json_file)
    
    
    # load pickle file
    info = {}
    with open(pickle_file, 'rb') as fp:
        pkldata = pickle.load(fp, encoding='utf-8')
        print(f"{dataset} utterance num in {pickle_file} is:  {len(pkldata.keys())}.")
    
    # wav scp
    wavscp_dct = {}
    with open(wavscp, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split()
            wavscp_dct[line[0]] = line[1]
    print(f"wav.scp utterance num: {len(wavscp_dct)}")
    
    
    results = []
    if ifmulticore is True:
        keylist = list(oridata["utts"].keys())
        pool = mp.Pool()
        job_args = [
            (
                i,
                oridata,
                pkldata,
                vdata,
                wavscp_dct,
                dataset,
            )
            for i in keylist
        ]
        results.extend(pool.map(product_helper, job_args))
    else:
        keylist = list(oridata["utts"].keys())
        for i in keylist:
            results.append(
                processing(
                    i,
                    oridata,
                    pkldata,
                    vdata,
                    wavscp_dct,
                    dataset,
                )
            )

    for i in range(len(results)):
        output["utts"].update(results[i])
    
    # import pdb; pdb.set_trace()
    print(f"Save to new data2json file's utterance num of {dataset} set: {len(results)}")
    savefilename = filename.replace(oridumpfile, dumpfile)
    if not os.path.exists(savefilename.replace(jsonname, "")):
        os.makedirs(savefilename.replace(jsonname, ""))
    with open(savefilename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    


# hand over parameter overview
# sys.argv[1] = dumpfile (str), Directory to save dump files
# sys.argv[2] = original dump file
# sys.argv[3] = pickle file
# sys.argv[4] = video dump file, just for video frame shape: vshape
# sys.argv[5] = wavscp file, for the wav path
# sys.argv[6] = dset(str), Which dataset
# sys.argv[7] = ifmulticore (boolean), If multi cpu processing should be used

redump(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    sys.argv[7],
)
