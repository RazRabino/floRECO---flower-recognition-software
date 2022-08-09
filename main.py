import os
# needed because of tensorFlow version changes
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import os.path
import tensorflow as tf
# needed because of tensorFlow gpu version is not used
tf.get_logger().setLevel('ERROR')
from keras.models import load_model
from PIL import Image, ImageOps, ImageTk
from datetime import datetime
import numpy as np
import os


def handle_load(images_list):
    # later will be used to check if image is already in the list
    buffered = [item for item in images_list.get(0, tk.END)]

    # only jpg and png image files are allowed
    filepath = askopenfilename(
        filetypes=[("image Files", "*.png"), ("image Files", "*.jpg")]
    )
    if not filepath:
        # if user cancel or exited loading screen we need to return to main
        return
    with open(filepath) as input_file:
        if filepath in buffered:
            # if filepath is already in "buffered" there is no need to add it again
            return

        # print to console information and insert the path to the filepath list
        print("file loaded: " + str(input_file).split("\'")[1])
        images_list.insert(tk.END, str(input_file).split("\'")[1])


def handle_delete(images_list):
    # if there is selected valid item in the path list delete it from list
    if images_list.curselection():
        print("listbox item line " + str(images_list.curselection()[0]) + " had been deleted.")
        images_list.delete(images_list.curselection()[0], last=None)


def handle_credits():
    # alert box with credits information
    messagebox.showinfo("Credits", "floRECO development credits:\n              "
                                   "Raz Rabino\n              "
                                   "Hadas Arbel")


def handle_identify(selected_image, images_list, deep_learning_output):
    # flowers labels classes
    labels = {0: "bird of paradise",
              1: "bougainvillea",
              2: "cyclamen",
              3: "dahlia",
              4: "daisy",
              5: "dandelion",
              6: "foxglove",
              7: "frangipani",
              8: "gentian",
              9: "geranium",
              10: "hibiscus",
              11: "iris",
              12: "lily",
              13: "marigold",
              14: "orchid",
              15: "passion flower",
              16: "petunia",
              17: "rose",
              18: "sunflower",
              19: "tulip",
              20: "wallflower",
              21: "watercress"
              }

    # for a chosen valid path in list, recognise it with the use of our model
    if images_list.curselection():
        model = load_model("model/floRECO_M.h5")

        # image convert to model input shape
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(selected_image["file_path"])
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # image recognition
        prediction = model.predict(data)

        # choose three best values and use them only if needed (also print to console the values)
        index = [i for i in range(22)]
        max_values = sorted(zip(prediction[0], index), reverse=True)[:3]
        if max_values[0][0] / 1 * 100 > 0.01:
            print(str("{:.2f}".format(max_values[0][0] / 1 * 100)) + "%", labels[max_values[0][1]])
        if max_values[1][0] / 1 * 100 > 0.01:
            print(str("{:.2f}".format(max_values[1][0] / 1 * 100)) + "%", labels[max_values[0][1]])
        if max_values[2][0] / 1 * 100 > 0.01:
            print(str("{:.2f}".format(max_values[2][0] / 1 * 100)) + "%", labels[max_values[2][1]])

        # update "deep_learning_output" label inside program gui with the best value
        deep_learning_output.configure(text="floRECO identify this flower photo as " + labels[max_values[0][1]] +
                                            ".\nin success rate of " + "{:.2f}".format(max_values[0][0] / 1 * 100) + "%"
                                       )


def show_selected(images_list, selected_image, image_file_show,
                  image_file_title, image_file_info, deep_learning_output):
    # for a chosen valid path in list, update program state
    if images_list.curselection():
        selected_image["file_path"] = images_list.get(images_list.curselection()[0], last=None)
        selected_image["file_size"] = str(os.path.getsize(selected_image["file_path"]))
        selected_image["file_created_date"] = str(
            datetime.fromtimestamp(os.path.getctime(selected_image["file_path"])).strftime('%Y-%m-%d %H:%M:%S'))
        selected_image["file_name"] = selected_image["file_path"].split("/")[-1][
                                      :(selected_image["file_path"].split("/"))[-1].find(".")]
        selected_image["file_type"] = os.path.splitext(selected_image["file_path"])[-1]
        print("user last selection: " + selected_image["file_path"])

        photo = 0
        # load image according to its type(jpg/png)
        if selected_image["file_path"].endswith(".png"):
            photo = tk.PhotoImage(file=selected_image["file_path"])
        elif selected_image["file_path"].endswith(".jpg"):
            image = Image.open(selected_image["file_path"])
            photo = ImageTk.PhotoImage(image)
        else:
            return photo

        # update gui state
        new_image = photo
        image_file_show.configure(image=new_image)
        image_file_show.image = new_image
        image_file_title.configure(text=selected_image["file_name"])
        image_file_info.configure(
            text="file size: " + selected_image["file_size"] + " bytes\nfile created in: " + selected_image[
                "file_created_date"] + "\nType Of File: " + selected_image["file_type"])

        deep_learning_output.configure(text="floRECO - flower recognition software"
                                            "\n\n"
                                            "load image from your computer using \"Load Image To List\" Button"
                                            "\n\n"
                                            "delete image from list using \"Delete Image From List\" \n(after "
                                            "selecting image in list). "
                                            "\n\n"
                                            "recognise flower in image using \"Identify Flower Type\" \n(after "
                                            "selecting image in list). "
                                            "\n\n"
                                            "save image recognition data to pc using \"Save Image Data To PC\" \n("
                                            "after selecting image in list\n "
                                            "and after that using recognition on that image - \nneed to be done one "
                                            "after one).")


def main():
    print("code is started to running.")

    window = tk.Tk()
    # starting point state data
    selected_image = {"file_path": "images/sample.png", "file_size": 0, "file_created_date": 0,
                      "file_name": "No Image Selected!", "file_type": ".png"}

    def title_show(e):
        # center program gui title
        title_pos = int(window.winfo_width() / 3.5)
        title_text = "floRECO".rjust(title_pos // 2)
        window.title(title_text)
        return e

    content = tk.Frame(
        master=window,
        width=1050,
        height=650,
        background="#bdbdbd"
    )
    content.pack(
        fill=tk.BOTH,
        side=tk.LEFT,
        expand=True
    )

    body = tk.Frame(
        master=content,
        width=1050,
        height=450
    )
    body.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    main_body = tk.Frame(
        master=body,
        width=650,
        height=450,
    )
    main_body.pack(
        fill=tk.BOTH,
        side=tk.LEFT,
        expand=True
    )

    image_file_title = tk.Label(
        master=main_body,
        text=selected_image["file_name"],
        pady=50
    )
    image_file_title.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=False
    )

    img = tk.PhotoImage(file=selected_image["file_path"])
    image_file_show = tk.Label(
        master=main_body,
        image=img,
        width=350,
        height=350
    )
    image_file_show.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=False
    )

    image_file_info = tk.Label(
        master=main_body,
        text="",
        pady=50
    )
    image_file_info.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=False
    )

    side_body = tk.Frame(
        master=body,
        height=450,
        width=400,
        background="#d6d6d6"
    )
    side_body.pack(
        fill=tk.BOTH,
        side=tk.RIGHT,
        expand=True
    )

    deep_learning_output = tk.Label(
        background="#d6d6d6",
        master=side_body,
        text="floRECO - flower recognition software"
             "\n\n"
             "load image from your computer using \"Load Image To List\" Button"
             "\n\n"
             "delete image from list using \"Delete Image From List\" \n(after selecting image in list)."
             "\n\n"
             "recognise flower in image using \"Identify Flower Type\" \n(after selecting image in list)."
             "\n\n"
             "save image recognition data to pc using \"Save Image Data To PC\" \n(after selecting image in list\n"
             "and after that using recognition on that image - \nneed to be done one after one)."
    )
    deep_learning_output.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=False
    )
    deep_learning_output.place(
        relx=.5,
        rely=.5,
        anchor="center"
    )

    footer = tk.Frame(
        master=content,
        width=1050,
        height=200
    )
    footer.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    main_footer = tk.Frame(
        master=footer,
        width=650,
        height=200,
        relief=tk.SUNKEN
    )
    main_footer.pack(
        fill=tk.BOTH,
        side=tk.LEFT,
        expand=True
    )

    images_list = tk.Listbox(
        background="#d9d9d9",
        foreground="black",
        master=main_footer,
        height=12,
        selectmode="SINGLE",
        selectbackground="#ffc108",
        bd=4
    )
    images_list.pack(
        side=tk.LEFT,
        fill=tk.BOTH,
        expand=True
    )
    images_list.bind('<<ListboxSelect>>',
                     lambda call: show_selected(images_list, selected_image, image_file_show, image_file_title,
                                                image_file_info, deep_learning_output))
    images_scrollbar = tk.Scrollbar(images_list)
    images_scrollbar.pack(
        side=tk.RIGHT,
        fill=tk.BOTH
    )

    # add scrollbar to images_list(images path list)
    images_list.config(yscrollcommand=images_scrollbar.set)
    images_scrollbar.config(command=images_list.yview)

    side_footer = tk.Frame(
        master=footer,
        width=400,
        height=200,
        relief=tk.RAISED
    )
    side_footer.pack(
        fill=tk.BOTH,
        side=tk.LEFT,
        expand=False
    )

    load_button = tk.Button(
        master=side_footer,
        text="Load Image To List",
        height=2,
        command=lambda: handle_load(images_list)
    )
    load_button.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    delete_button = tk.Button(
        master=side_footer,
        text="Delete Image From List",
        height=2,
        command=lambda: handle_delete(images_list)
    )
    delete_button.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    identify_button = tk.Button(
        master=side_footer,
        text="Identify Flower Type",
        height=2,
        command=lambda: handle_identify(selected_image, images_list, deep_learning_output)
    )
    identify_button.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    credits_button = tk.Button(
        master=side_footer,
        text="Credits",
        height=2,
        command=handle_credits
    )
    credits_button.pack(
        fill=tk.BOTH,
        side=tk.TOP,
        expand=True
    )

    # set icon for gui window
    icon = tk.PhotoImage(file="images/icon.png")
    window.iconphoto(False, icon)

    # call for center gui title update every window size change
    window.bind("<Configure>", title_show)
    window.eval("tk::PlaceWindow . center")

    # tk mainloop
    window.mainloop()

    print("code is finished running.")


if __name__ == '__main__':
    main()
