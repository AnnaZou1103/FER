import tkinter as tk
import tkinter.filedialog
import time
import os
from process_file import process_file


# Set process function of each button
def set_image():
    entry_upload_path['state'] = 'normal'
    btn_upload['state'] = 'normal'


def set_video():
    entry_upload_path['state'] = 'normal'
    btn_upload['state'] = 'normal'


def set_upload_path():
    path_ = tkinter.filedialog.askopenfilename()
    upload_path.set(path_)


def select_save_path():
    path_ = tkinter.filedialog.askdirectory()
    save_path.set(path_ + '/')


def process():
    start = time.time()
    process_file(upload_path.get(), save_path.get(), input_type.get())
    end = time.time()
    seconds = end - start

    hour = int(seconds / 3600)
    minute = int(seconds % 3600 / 60)
    second = int(seconds % 60)

    topw, toph = 310, 100
    ctpx = int(sw / 2 - topw / 2)
    ctpy = int(sh / 2 - toph / 2)
    top = tk.Toplevel(window)
    top.geometry(f"{topw}x{toph}+{ctpx}+{ctpy}")
    top.title('Notice')
    finish_top = tk.Label(top, font=('Arial', 14),
                          text='Process completed in ' + str(hour) + 'h ' + str(minute) + "m " + str(second) + 's. ')
    finish_top.place(x=10, y=10)
    btn_ok = tk.Button(top, font=('Arial', 14), text="OK", width=8, command=top.destroy)
    btn_ok.place(x=100, y=50)
    entry_upload_path.delete(0, 'end')


if __name__ == '__main__':
    window = tk.Tk()
    window.title('Facial Emotion Recognition')

    sw = window.winfo_screenwidth()
    sh = window.winfo_screenheight()
    ww = 650  # Set window width
    wh = 210  # Set window height
    x = int((sw - ww) / 2)
    y = int((sh - wh) / 2)
    window.geometry(f"{ww}x{wh}+{x}+{y}")

    # Set input type
    input_type = tk.StringVar()
    label_input = tk.Label(window, font=('Arial', 14), text='Please select the input type:')
    label_input.place(x=10, y=10)

    input_type.set('Image')
    r_btn_input_1 = tk.Radiobutton(window, font=('Arial', 14), text='Image', variable=input_type, value='Image',
                                   command=set_image)
    r_btn_input_1.place(x=200, y=10)

    r_btn_input_2 = tk.Radiobutton(window, font=('Arial', 14), text='Video', variable=input_type, value='Video',
                                   command=set_video)
    r_btn_input_2.place(x=350, y=10)

    # Set upload path
    upload_path = tk.StringVar()
    label_upload = tk.Label(window, text="Input Path:", font=('Arial', 14))
    label_upload.place(x=10, y=60)
    entry_upload_path = tk.Entry(window, font=('Arial', 14), textvariable=upload_path)
    entry_upload_path.place(x=120, y=60, width=350)
    btn_upload = tk.Button(window, text="Open", font=('Arial', 14), width=10, command=set_upload_path)
    btn_upload.place(x=480, y=60)

    # Set output path
    default_save_path = 'output/processed_media/'
    tk.Label(window, text='Save Path:', font=('Arial', 14)).place(x=10, y=110)
    if not os.path.exists(default_save_path):
        os.makedirs(default_save_path)
    save_path = tk.StringVar()
    save_path.set(default_save_path)
    entry_save_path = tk.Entry(window, font=('Arial', 14), textvariable=save_path)
    entry_save_path.place(x=120, y=110, width=350)
    btn_save = tk.Button(window, text="Open", font=('Arial', 14), width=10, command=select_save_path)
    btn_save.place(x=480, y=110)

    # Set buttons
    btn_process = tk.Button(window, font=('Arial', 14), text="Execute", width=10, command=process)
    btn_process.place(x=180, y=160)

    btn_quit = tk.Button(window, font=('Arial', 14), text="Quit", width=10, command=window.quit)
    btn_quit.place(x=340, y=160)

    window.mainloop()
