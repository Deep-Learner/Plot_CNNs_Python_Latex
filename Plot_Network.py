import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

architecture = [{"width" : 416, "height" : 416, "filter" : 3, "kernel_size" : 3},
                {"width" : 208, "height" : 208, "filter" : 32, "kernel_size" : 3},
                {"width" : 104, "height" : 104, "filter" : 64, "kernel_size" : 3},
                {"width" : 52, "height" : 52, "filter" : 128, "kernel_size" : 3},
                {"width" : 26, "height" : 26, "filter" : 256, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 512, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 3072, "kernel_size" : 3},
                {"width" : 13, "height" : 13, "filter" : 1000, "kernel_size" : 3},
                ]

"""
for elem in architecture:
    elem["width"] *= 3
    elem["height"] *= 3
"""

img_width = 1920# 1600
img_height = 1080# 900
last_diag_width, last_diag_height = 3, 3
node_counter = 0


output_path = "\\Latex_Files\\Example.tex"

if not os.path.isdir(os.getcwd() + os.sep.join(output_path.split(os.sep)[:-1])):
    os.makedirs(os.getcwd() + os.sep.join(output_path.split(os.sep)[:-1]))


def draw_box(d, img=None, scale=1, x_shift=0, y_shift=0, plot_kernel=False):
    global last_diag_width, last_diag_height, node_counter, output_path
    print_latex_code = True

    thickness=1

    if img is None:
        img = np.ones((img_height, img_width, 3), dtype=float) * 255

    top_left = (x_shift,y_shift)
    bottom_left = (x_shift,y_shift+d["height"])
    top_right = (x_shift+int(np.log2(d["filter"])*10),y_shift)
    bottom_right = (x_shift+int(np.log2(d["filter"])*10),y_shift+d["height"])

    latex_scale_factor = 50
    latex_width = round(d["width"] / latex_scale_factor, 2)
    latex_height = round(d["height"] / latex_scale_factor, 2)
    top_left_latex = (round(x_shift / latex_scale_factor, 2), latex_height)
    top_right_latex = (round((x_shift + np.log2(d["filter"]) * 10) / latex_scale_factor, 2), latex_height)
    bottom_right_latex = (round((x_shift + np.log2(d["filter"]) * 10) / latex_scale_factor, 2), 0)
    bottom_left_latex = (round(x_shift / latex_scale_factor, 2), 0)

    cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right,color=(0,0,0), thickness=thickness)

    length_diag = np.sqrt((np.array(top_left) - np.array([top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))])).sum())

    # print("np.array(top_left) =", np.array(top_left))
    # print("[top_left[0] - int(d[width] / 2), top_left[1] - int(d[height] / 2)] =", np.array([top_left[0]-int(d["width"]/2),top_left[1]-int(d["height"]/2)]))

    # Lines which go diagonal left backwards
    cv2.line(img=img, pt1=top_left, pt2=(top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))), color=(0, 0, 0), thickness=thickness)
    cv2.line(img=img, pt1=top_right, pt2=(top_right[0]-int(d["width"]/(2*np.sqrt(2))),top_right[1]-int(d["height"]/(2*np.sqrt(2)))), color=(0, 0, 0), thickness=thickness)
    cv2.line(img=img, pt1=bottom_left, pt2=(bottom_left[0]-int(d["width"]/(2*np.sqrt(2))),bottom_left[1]-int(d["height"]/(2*np.sqrt(2)))), color=(0, 0, 0), thickness=thickness)

    # Horizontal lines which are in the back
    cv2.line(img=img, pt1=(top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))), pt2=(top_right[0]-int(d["width"]/(2*np.sqrt(2))),top_right[1]-int(d["height"]/(2*np.sqrt(2)))), color=(0, 0, 0), thickness=thickness)
    cv2.line(img=img, pt1=(top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))), pt2=(bottom_left[0]-int(d["width"]/(2*np.sqrt(2))),bottom_left[1]-int(d["height"]/(2*np.sqrt(2)))), color=(0, 0, 0), thickness=thickness)

    # Draw Kernels:
    if plot_kernel:
        kernel_center_left = np.array(bottom_left) - 0.5 * (np.array(bottom_left) - np.array([top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))]))
        kernel_center_left = (int(kernel_center_left[0]), int(kernel_center_left[1]))
        # kernel_center_right = np.array(bottom_right) - 0.5 * (np.array(bottom_right) - np.array([bottom_right[0] - int(d["width"] / (2 * np.sqrt(2))), bottom_right[1] - int(d["height"] / (2 * np.sqrt(2)))]))
        # kernel_center_right = (int(kernel_center_right[0]), int(kernel_center_right[1]))
        kernel_center_right = (kernel_center_left[0]+bottom_right[0]-bottom_left[0], int(kernel_center_left[1]))
        # cv2.circle(img, kernel_center_left, radius=2, color=(0, 0, 0))
        # cv2.circle(img, kernel_center_right, radius=2, color=(0, 0, 0))

        # print("np.sqrt(d[height]**2 + d[width]**2) =", np.sqrt(d["height"]**2 + d["width"]**2))

        # plotted_kernel_size = (np.log2(np.log2(np.sqrt(d["height"]**2 + d["width"]**2)))) # int(bottom_left[1] - top_left[1]) # int(np.sqrt(d["height"])+(d["kernel_size"]+2)/(np.sqrt(2)))
        plotted_kernel_size = np.sqrt(np.sqrt(d["height"] + d["width"])) # int(bottom_left[1] - top_left[1]) # int(np.sqrt(d["height"])+(d["kernel_size"]+2)/(np.sqrt(2)))
        kernel_front_top_left = (kernel_center_left[0] + plotted_kernel_size/(2*np.sqrt(2)), kernel_center_left[1] - plotted_kernel_size)
        kernel_front_bottom_right = (kernel_center_right[0] + plotted_kernel_size/(2*np.sqrt(2)), kernel_center_right[1] + 2*plotted_kernel_size)
        kernel_front_top_right = (kernel_front_bottom_right[0], kernel_front_top_left[1])
        kernel_front_bottom_left = (kernel_front_top_left[0], kernel_front_bottom_right[1])
        cv2.rectangle(img=img, pt1=(int(kernel_front_top_left[0]), int(kernel_front_top_left[1])), pt2=(int(kernel_front_bottom_right[0]), int(kernel_front_bottom_right[1])), color=(0, 0, 0), thickness=thickness)

        # print("high box =", kernel_front_bottom_left[1] - kernel_front_top_left[1])
        # print("high box =", kernel_front_bottom_left[1] - kernel_front_top_left[1])

        # Lines which go diagonal left backwards
        # cv2.line(img=img, pt1=kernel_front_top_left, pt2=(kernel_front_top_left[0] - plotted_kernel_size, kernel_front_top_left[1] - plotted_kernel_size), color=(0, 0, 0))
        # cv2.line(img=img, pt1=kernel_front_top_right, pt2=(kernel_front_top_right[0] - plotted_kernel_size, kernel_front_top_right[1] - plotted_kernel_size), color=(0, 0, 0))
        # cv2.line(img=img, pt1=kernel_front_bottom_left, pt2=(kernel_front_bottom_left[0] - plotted_kernel_size, kernel_front_bottom_left[1] - plotted_kernel_size), color=(0, 0, 0))


        # Horizontal lines which are in the back
        # cv2.line(img=img, pt1=(kernel_front_top_left[0] - plotted_kernel_size, kernel_front_top_left[1] - plotted_kernel_size), pt2=(kernel_front_top_right[0] - plotted_kernel_size, kernel_front_top_right[1] - plotted_kernel_size), color=(0, 0, 0))
        # cv2.line(img=img, pt1=(kernel_front_top_left[0] - plotted_kernel_size, kernel_front_top_left[1] - plotted_kernel_size), pt2=(kernel_front_bottom_left[0] - plotted_kernel_size, kernel_front_bottom_left[1] - plotted_kernel_size), color=(0, 0, 0))

        # Lines which go diagonal left backwards
        cv2.line(img=img, pt1=(int(kernel_front_top_left[0]), int(kernel_front_top_left[1])), pt2=(int(kernel_front_top_left[0] - plotted_kernel_size), int(kernel_front_top_left[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)
        cv2.line(img=img, pt1=(int(kernel_front_top_right[0]), int(kernel_front_top_right[1])),pt2=(int(kernel_front_top_right[0] - plotted_kernel_size), int(kernel_front_top_right[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)
        cv2.line(img=img, pt1=(int(kernel_front_bottom_left[0]), int(kernel_front_bottom_left[1])),pt2=(int(kernel_front_bottom_left[0] - plotted_kernel_size), int(kernel_front_bottom_left[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)

        # Horizontal lines which are in the back
        cv2.line(img=img, pt1=(int(kernel_front_top_left[0] - plotted_kernel_size), int(kernel_front_top_left[1] - plotted_kernel_size)),pt2=(int(kernel_front_top_right[0] - plotted_kernel_size), int(kernel_front_top_right[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)
        cv2.line(img=img, pt1=(int(kernel_front_top_left[0] - plotted_kernel_size), int(kernel_front_top_left[1] - plotted_kernel_size)),pt2=(int(kernel_front_bottom_left[0] - plotted_kernel_size), int(kernel_front_bottom_left[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)

        # cv2.circle(img, kernel_center, radius=2, color=(0,0,0))
        # cv2.circle(img, bottom_left, radius=2, color=(255, 0, 0), thickness=3)
        # cv2.circle(img, (top_left[0]-int(d["width"]/(2*np.sqrt(2))),top_left[1]-int(d["height"]/(2*np.sqrt(2)))), radius=2, color=(0, 255, 0), thickness=3)
        """
        kernel_top_left = tuple(np.array(np.array(kernel_center) - np.array([(d["kernel_size"]+1)/2,(d["kernel_size"]+1)/2]), dtype=int))
        kernel_bottom_right = tuple(np.array(np.array(kernel_center) + np.array([(d["kernel_size"]+1)/2,(d["kernel_size"]+1)/2]), dtype=int))
        print("kernel_top_left =", kernel_top_left)
        kernel_top_right = (int(kernel_bottom_right[0]), int(kernel_top_left[1]))
        # cv2.circle(img=img, center=(int(kernel_bottom_right[0]), int(kernel_bottom_right[1])), radius=4, color=(0, 0, 0), thickness=13)
        cv2.rectangle(img=img, pt1=kernel_top_left, pt2=kernel_bottom_right, color=(0, 0, 0), thickness=1)
        """

    """
    cv2.line(img=img,
             pt1=(top_left[0] - int(d["width"] / (2 * np.sqrt(2))), top_left[1] - int(d["height"] / (2 * np.sqrt(2)))),
             pt2=(bottom_left[0] - int(d["width"] / (2 * np.sqrt(2))), bottom_left[1] - int(d["height"] / (2 * np.sqrt(2)))),
             color=(0, 0, 0))
    """

    if print_latex_code:
        print("\n%New Layer begins:", file=open(os.getcwd() + output_path, "a"))
        print("\\draw (" + str(top_left_latex[0]) + "," + str(top_left_latex[1]) + ") -- (" +
              str(top_right_latex[0]) + "," + str(top_right_latex[1]) + ") -- (" +
              str(bottom_right_latex[0]) + "," + str(bottom_right_latex[1]) + ") -- (" +
              str(bottom_left_latex[0]) + "," + str(bottom_left_latex[1]) + ") -- (" +
              str(top_left_latex[0]) + "," + str(top_left_latex[1]) + "); \t%rectangle in the front with the order: top_left, top_right, bottom_right, bottom_left, top_left",
              file=open(os.getcwd() + output_path, "a"))
        # """
        # print("length_diag =", length_diag)
        """
        if length_diag < 0:
            last_diag_width = round(latex_width / (2 * np.sqrt(2)), 2)
            last_diag_height = round(latex_height / (2 * np.sqrt(2)), 2)
        """
        last_diag_width = round(latex_width / (2 * np.sqrt(2)), 2)
        last_diag_height = round(latex_height / (2 * np.sqrt(2)), 2)

        # Lines which go diagonal left backwards
        print("\\draw (" + str(top_left_latex[0]) + "," + str(top_left_latex[1]) + ") -- (" +
              str(top_left_latex[0] - last_diag_width) + "," + str(top_left_latex[1] + last_diag_height) + "); \t%diagonal line from top_left backwards", file=open(os.getcwd() + output_path, "a"))
        print("\\draw (" + str(top_right_latex[0]) + "," + str(top_right_latex[1]) + ") -- (" +
              str(top_right_latex[0] - last_diag_width) + "," + str(top_right_latex[1] + last_diag_height) + "); \t%diagonal line from top_right backwards", file=open(os.getcwd() + output_path, "a"))
        print("\\draw (" + str(bottom_left_latex[0]) + "," + str(bottom_left_latex[1]) + ") -- (" +
              str(bottom_left_latex[0] - last_diag_width) + "," + str(bottom_left_latex[1] + last_diag_height) + "); \t%diagonal line from bottom_left backwards", file=open(os.getcwd() + output_path, "a"))

        # Horizontal and vertical lines which are in the back
        print("\\draw (" + str(top_left_latex[0] - last_diag_width) + "," + str(top_left_latex[1] + last_diag_height) + ") -- (" +
              str(top_right_latex[0] - last_diag_width) + "," + str(top_right_latex[1] + last_diag_height) + "); \t%horizontal line from top_left to top_right", file=open(os.getcwd() + output_path, "a"))
        print("\\draw (" + str(top_left_latex[0] - last_diag_width) + "," + str(top_left_latex[1] + last_diag_height) + ") -- (" +
              str(bottom_left_latex[0] - last_diag_width) + "," + str(bottom_left_latex[1] + last_diag_height) + "); \t%vertical line from top_left to bottom_left", file=open(os.getcwd() + output_path, "a"))


        text_size = int(np.log2(d["width"])) + 2
        text_margin_x = - 0.06 * text_size # - 0.07 * text_size
        text_margin_y = - 0.02 * text_size
        # print("text_size =", text_size)
        text_width_pos = (bottom_left_latex[0] - 0.5*last_diag_width + text_margin_x, bottom_left_latex[1] + 0.5*last_diag_height + text_margin_y)
        text_height_pos = (top_left_latex[0] - last_diag_width + text_margin_x, bottom_left_latex[1]+last_diag_height+0.5*latex_height - text_margin_y)
        text_filter_pos = (bottom_left_latex[0] - 0.5*(bottom_left_latex[0] - bottom_right_latex[0]), + bottom_left_latex[1] + 0.5*(bottom_left_latex[1] - bottom_right_latex[1]) + 2*text_margin_y)
        # print("\\node(layer" + str(node_counter) + ") at " + str(text_bottom_pos) + " {$\\cdot$};")
        print("\\node(layer" + str(node_counter) + "w) at " + str(text_width_pos) + " {\\fontsize{"+str(text_size)+"}{"+str(text_size)+"}$"+str(d["width"])+"$};", file=open(os.getcwd() + output_path, "a"))
        print("\\node(layer" + str(node_counter) + "h) at " + str(text_height_pos) + " {\\fontsize{"+str(text_size)+"}{"+str(text_size)+"}$"+str(d["height"])+"$};", file=open(os.getcwd() + output_path, "a"))
        print("\\node(layer" + str(node_counter) + "f) at " + str(text_filter_pos) + " {\\fontsize{"+str(text_size)+"}{"+str(text_size)+"}$"+str(d["filter"])+"$};", file=open(os.getcwd() + output_path, "a"))

        if plot_kernel:
            kernel_center_left_latex = np.array(bottom_left_latex) + 0.5 * (np.array([top_left_latex[0] - last_diag_width, top_left_latex[1] + last_diag_height])-np.array(bottom_left_latex))
            # print("kernel_center_left_latex =", kernel_center_left_latex)
            kernel_center_left_latex = (kernel_center_left_latex[0], kernel_center_left_latex[1])
            kernel_center_right_latex = (kernel_center_left_latex[0] + bottom_right_latex[0] - bottom_left_latex[0], kernel_center_left_latex[1])

            plotted_kernel_size_latex = 0.08*np.sqrt(latex_height+latex_width) # np.max(0.0, 0.5*np.sqrt(int(latex_height)**2 + int(latex_width)**2))  # int(bottom_left_latex[1] - top_left_latex[1]) # int(np.sqrt(d["height"])+(d["kernel_size"]+2)/(np.sqrt(2)))
            print("%plotted_kernel_size_latex =", plotted_kernel_size_latex)
            kernel_front_top_left_latex = (kernel_center_left_latex[0] + plotted_kernel_size_latex / (2 * np.sqrt(2)), kernel_center_left_latex[1] + plotted_kernel_size_latex)
            kernel_front_bottom_right = (kernel_center_right_latex[0] + plotted_kernel_size_latex / (2 * np.sqrt(2)), kernel_center_right_latex[1] -  np.sqrt(2) * plotted_kernel_size_latex)
            kernel_front_top_right = (kernel_front_bottom_right[0], kernel_front_top_left_latex[1])
            kernel_front_bottom_left_latex = (kernel_front_top_left_latex[0], kernel_front_bottom_right[1])

            # Horizontal lines which are in the back
            # cv2.line(img=img, pt1=(int(kernel_front_top_left[0] - plotted_kernel_size), int(kernel_front_top_left[1] - plotted_kernel_size)),pt2=(int(kernel_front_top_right[0] - plotted_kernel_size), int(kernel_front_top_right[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)
            # cv2.line(img=img, pt1=(int(kernel_front_top_left[0] - plotted_kernel_size), int(kernel_front_top_left[1] - plotted_kernel_size)),pt2=(int(kernel_front_bottom_left[0] - plotted_kernel_size), int(kernel_front_bottom_left[1] - plotted_kernel_size)), color=(0, 0, 0), thickness=thickness)

            print("\\draw (" + str(round(kernel_front_top_left_latex[0], 3)) + "," + str(round(kernel_front_top_left_latex[1], 3)) + ") -- (" +
                  str(round(kernel_front_top_right[0], 3)) + "," + str(round(kernel_front_top_right[1], 3)) + ") -- (" +
                  str(round(kernel_front_bottom_right[0], 3)) + "," + str(round(kernel_front_bottom_right[1], 3)) + ") -- (" +
                  str(round(kernel_front_bottom_left_latex[0], 3)) + "," + str(round(kernel_front_bottom_left_latex[1], 3)) + ") -- (" +
                  str(round(kernel_front_top_left_latex[0], 3)) + "," + str(round(kernel_front_top_left_latex[1], 3)) + ");", file=open(os.getcwd() + output_path, "a"))

            # Lines which go diagonal left backwards
            print("\\draw (" + str(round(kernel_front_top_left_latex[0], 3)) + "," + str(round(kernel_front_top_left_latex[1], 3)) + ") -- (" +
                  str(round(kernel_front_top_left_latex[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_top_left_latex[1] + plotted_kernel_size_latex, 3)) + "); \t%diagonal line from top_left backwards", file=open(os.getcwd() + output_path, "a"))
            print("\\draw (" + str(round(kernel_front_top_right[0], 3)) + "," + str(round(kernel_front_top_right[1], 3)) + ") -- (" +
                  str(round(kernel_front_top_right[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_top_right[1] + plotted_kernel_size_latex, 3)) + "); \t%diagonal line from top_right backwards", file=open(os.getcwd() + output_path, "a"))
            print("\\draw (" + str(round(kernel_front_bottom_left_latex[0], 3)) + "," + str(round(kernel_front_bottom_left_latex[1], 3)) + ") -- (" +
                  str(round(kernel_front_bottom_left_latex[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_bottom_left_latex[1] + plotted_kernel_size_latex, 3)) + "); \t%diagonal line from bottom_left backwards", file=open(os.getcwd() + output_path, "a"))

            # Horizontal lines which are in the back
            print("\\draw (" + str(round(kernel_front_top_left_latex[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_top_left_latex[1] + plotted_kernel_size_latex, 3)) + ") -- (" + str(round(kernel_front_top_right[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_top_right[1] + plotted_kernel_size_latex, 3)) + "); \t%horizontal from top_left to top_right", file=open(os.getcwd() + output_path, "a"))
            print("\\draw (" + str(round(kernel_front_top_left_latex[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_top_left_latex[1] + plotted_kernel_size_latex, 3)) + ") -- (" + str(
                round(kernel_front_bottom_left_latex[0] - plotted_kernel_size_latex, 3)) + "," + str(
                round(kernel_front_bottom_left_latex[1] + plotted_kernel_size_latex, 3)) + "); \t%horizontal from top_left to bottom_left", file=open(os.getcwd() + output_path, "a"))

            # print("\\node(v"+str(node_counter)+") at " + str(kernel_center_left_latex) + " {$\\cdot$};")
            # print("\\node(v"+str(10+node_counter)+") at " + str(kernel_center_right_latex) + " {$\\cdot$};")


            node_counter += 1
    return img

def draw_net(net):
    global output_path
    x_shift = 200
    y_shift = 700
    last_filter = 0
    img = None
    # y_shift = int((img_height - net[0]["height"]))


    f = open(os.getcwd() + output_path, "w+")
    f.close()

    print("\\documentclass[a4paper,english]{scrbook}", file=open(os.getcwd() + output_path, "a"))
    print("\\usepackage{tikz}", file=open(os.getcwd() + output_path, "a"))
    print("\\begin{document}", file=open(os.getcwd() + output_path, "a"))
    print("\\begin{center}", file=open(os.getcwd() + output_path, "a"))
    print("\\begin{tikzpicture}[>=latex, line width=0.01mm, inner sep=0pt, scale=0.53]", file=open(os.getcwd() + output_path, "a"))
    for i, layer in enumerate(net):
        # img = draw_box(layer, img=img, x_shift=(i+1)*10+40*int(np.log2(layer["filter"])), y_shift=100)
        # img = draw_box(layer, img=img, x_shift=100*int(np.log2(layer["filter"])), y_shift=100)
        # img = draw_box(layer, img=img, x_shift=(i+2)*180, y_shift=700-layer["height"])
        # print("\t\t\t(i+1)*100 =", (i+1)*100)
        # print("\t\t\t100*int(np.log2(layer[filter])) =", 100*int(np.log2(layer["filter"])))
        # print("\t\t\tint(0.2 * np.log2(layer[width])) =", int(0.2 * np.log2(layer["width"])))
        # print("\t\t\t(i+1)*100+100*int(np.log2(layer[filter]))+int(0.0 * np.log2(layer[width])) =", (i+1)*100+100*int(np.log2(layer["filter"]))+int(0.0 * np.log2(layer["width"])))
        # img = draw_box(layer, img=img, x_shift=(i+1)*100+100*int(np.log2(layer["filter"]))+int(0.0 * np.log2(layer["width"])), y_shift=700-layer["height"])
        x_shift += 22 * int(np.sqrt(layer["width"]) / (2 * np.sqrt(2))) + 11 *last_filter + 40
        # x_shift += 50 * int(np.sqrt(layer["width"]) / (2 * np.sqrt(2))) + 11 *last_filter + 20        # for matplotlib and large shifts
        # print("\t\tx_shift =", x_shift)
        img = draw_box(layer, img=img, x_shift=x_shift, y_shift=y_shift-layer["height"], plot_kernel= (True if i < len(net)-1 else False))
        last_filter = int(np.log2(layer["filter"]))

    print("\\end{tikzpicture}", file=open(os.getcwd() + output_path, "a"))
    print("\\end{center}", file=open(os.getcwd() + output_path, "a"))
    print("\\end{document}", file=open(os.getcwd() + output_path, "a"))

    cv2.imshow("Test", img)
    # cv2.imwrite("Test.png", img, dpi=100)
    # plt.figure(figsize=(8, 8))

    """
    img = np.array(img, dtype=np.uint8)
    plt.figure(figsize=(16,9))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig("Test.png", dpi=1000)
    plt.close()
    """
    cv2.waitKey()

draw_net(architecture)