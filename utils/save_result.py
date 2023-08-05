from pathlib import Path
import glob
import re

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def save_resul(target_path,name,exist_ok,save_txt):
    save_dir = Path(increment_path(Path(target_path) / name, exist_ok=exist_ok)) 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    return save_dir
