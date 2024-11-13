from plyer import filechooser
import os
def file_chooser():
    extensions = [
    '.mp4',
    '.avi',
    '.mov',
    '.wmv',
    '.flv',
    '.mkv',
    '.webm',
    '.mpeg',
    '.mpg',
    '.3gp',
    '.m4v',
    '.ts',
    '.f4v'
]
    vid = filechooser.open_file()
    if vid:
        file_path = vid[0]
        if any(file_path.lower().endswith(extension) for extension in extensions):
            print(f"File valid format {file_path}")
        else: 
            print(f"File format must be {', '.join(extensions)}")
    else:
        print('No file')
    list = file_path.split('\\')
    print(list[-1])
    project_directory = os.path.dirname(os.path.abspath(__file__))
    print(project_directory+list[-1])

file_chooser()
