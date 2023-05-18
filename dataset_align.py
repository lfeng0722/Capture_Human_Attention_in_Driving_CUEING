import os

def namechange(gaze_path):
    list = os.listdir(gaze_path)
    for k in list:
        new_name=k.replace('_','_pure_hm_')
        new_name_all = gaze_path + '/' + new_name
        name = gaze_path + '/' + k
        os.rename(name, new_name_all)
def gaze2ori(gaze_path):
    list =os.listdir(gaze_path)

    for k in list:
        new_name=k.replace('_pure_hm_','_')
        name = gaze_path+ '/' +k
        new_name_all=gaze_path+ '/' +new_name
        os.rename(name,new_name_all)


def delete(ori_path,gaze_path):
    list1 = os.listdir(ori_path)
    list2 = os.listdir(gaze_path)
    for j in list2:
        if j not in list1:
            os.remove(gaze_path+'/'+j)

def re(gaze_path):
    list = os.listdir(gaze_path)
    for k in list:
        new_name = k.replace('_','_pure_hm_')
        name = gaze_path + '/' + k
        new_name_all = gaze_path + '/' + new_name
        os.rename(name, new_name_all)


if __name__ == "__main__":
    subset = 'training'
    gaze_path = f'DADA/{subset}/gazemap_images'
    ori_path = f'BDDA_unfiltered/{subset}/camera_images'
    namechange(gaze_path)
    # gaze2ori(gaze_path)
    # delete(ori_path,gaze_path)
    # re(gaze_path)