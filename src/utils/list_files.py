from pathlib import Path
import sys
from typing import List, Union
from os.path import join


def menu_selection(paths: Union[Path, List[Path]]) -> List[Path]:
    """
    Ask user to select directiories from the current folder
    """
    if not isinstance(paths, list):
        paths = list(paths.iterdir())
    paths.sort()
    print()
    for i, dir_ in enumerate(paths):
        printable = join(*dir_.parts[-2:])
        print(f"{i} - {printable} \n")

    print(f"{i+1} - All \n")

    print("Please enter corresponding digits (space separated)? \n")
    str_ans = str(input()).split()
    selected_digits = [int(x) for x in str_ans]
    if (i + 1) in selected_digits:
        return paths
    else:
        return [paths[i] for i in selected_digits]


if __name__ == '__main__':

    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    out_file = sys.argv[3]

    in_path = Path(in_folder)
    if "output" in out_folder:
        print("From which experiment to restore? \n")
        selected_dir = menu_selection(in_path)
        assert len(selected_dir) == 1, "Please select only 1 experiment"
        in_path = Path(selected_dir[0])

    print("Which folders would you like to select? \n")
    src_line = menu_selection(in_path)
    src_line = list(map(lambda x: str(x), src_line))  # type: ignore[arg-type]
    if isinstance(src_line, list):
        src_line = ' '.join(src_line)  # type: ignore[arg-type]
    with open(out_file, 'w') as f:
        f.write(src_line + "\n")  # type: ignore[arg-type]

    if "archive" in out_folder:
        print("What name for the experiment? \n")
        exp_name = str(input())
        archive_path = Path(out_folder) / in_path.stem / exp_name
        archive_path.mkdir(parents=True, exist_ok=True)

        with open(out_file, 'a') as f:
            f.write(str(archive_path))


