import customtkinter as ctk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from tkinter import filedialog
from scipy.ndimage import uniform_filter, median_filter

ctk.set_appearance_mode("light")


class Web(ctk.CTk):
    def enhance_colors(self, factor=1.2):
        if self.image is not None:
            img = self.image.astype(np.float32)
            img *= factor
            img = np.clip(img, 0, 255).astype(np.uint8)
            self.display_images(self.image, img, "Original", "Enhanced Colors")

    def __init__(self):
        super().__init__()
        self.geometry("1024x720")
        self.title("Image Processing App")
        self.image = None
        self.processed_image = None
        self.fig = None
        self.canvas = None
        self.threshold = 50  # Default threshold for outlier method

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Main display frame
        self.display_frame = ctk.CTkFrame(self, fg_color="#FFFCEF")
        self.display_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        # Sidebar frame
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=260, corner_radius=0, fg_color="#FFFCEF")
        self.sidebar_frame.grid(row=0, column=0, rowspan=10, sticky="nsew")

        # Load image button
        self.load_button = ctk.CTkButton(self, text="Load Image", command=self.load_image, font=("arial", 20, "bold"),
                                         fg_color="#008585", width=200, height=40, hover_color="#004343")
        self.load_button.grid(column=1, padx=20, row=0)

        # Point Operations Section
        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, text="Point operation", fg_color="#008585",
                                              command=self.toggle_point_menu, font=("arial", 18), width=200, height=40,
                                              hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\settings-sliders.png")))
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.point_menu_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.point_menu_frame.grid(row=2, column=0, padx=20, sticky="ew", pady=10)
        self.point_menu_frame.grid_remove()

        self.slider = ctk.CTkSlider(self.point_menu_frame, from_=0, to=100, button_color="#74a892",
                                    button_hover_color="#254e40")
        self.slider.grid(row=0, column=0, sticky="ew", pady=2)

        self.addi_bt = ctk.CTkButton(self.point_menu_frame, text="addition", command=self.apply_add_brightness,
                                     width=70, height=25, fg_color="#74a892",
                                     corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.addi_bt.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        self.sub_bt = ctk.CTkButton(self.point_menu_frame, text="subtraction", command=self.apply_subtract_brightness,
                                    width=70, height=25, fg_color="#74a892",
                                    corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.sub_bt.grid(row=2, column=0, sticky="ew", pady=2, padx=10)

        self.div_bt = ctk.CTkButton(self.point_menu_frame, text="division", command=self.apply_divide_image, width=70,
                                    height=25,
                                    fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.div_bt.grid(row=3, column=0, sticky="ew", pady=2, padx=10)

        self.comp_bt = ctk.CTkButton(self.point_menu_frame, text="complement", command=self.apply_complement_image,
                                     width=70, height=25,
                                     fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.comp_bt.grid(row=4, column=0, sticky="ew", pady=2, padx=10)

        # Color Operations Section
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, text="Color image operation", font=("arial", 18),
                                              fg_color="#008585", width=200, height=40, command=self.toggle_color_menu,
                                              hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\color-palette.png")))
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.color_menu_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.color_menu_frame.grid(row=4, column=0, padx=20, sticky="ew", pady=10)
        self.color_menu_frame.grid_remove()

        self.color_bt1 = ctk.CTkButton(self.color_menu_frame, text="Changing lighting color",
                                       command=self.apply_change_red,
                                       width=70, height=25, fg_color="#74a892", corner_radius=50, hover_color="#254e40",
                                       font=("arial", 14))
        self.color_bt1.grid(row=0, column=0, sticky="ew", pady=2, padx=10)

        self.color_bt2 = ctk.CTkButton(self.color_menu_frame, text="Swapping channels",
                                       command=self.apply_swap_red_green, width=70, height=25,
                                       fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.color_bt2.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        self.color_bt3 = ctk.CTkButton(self.color_menu_frame, command=self.apply_eliminate_red,
                                       text="Eliminating color channels", width=70, height=25,
                                       fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.color_bt3.grid(row=2, column=0, sticky="ew", pady=2, padx=10)

        # Histogram Section
        self.sidebar_button_3 = ctk.CTkButton(self.sidebar_frame, text="Image histogram", font=("arial", 18),
                                              fg_color="#008585", width=200, height=40, command=self.toggle_histo_menu,
                                              hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\bar-chart.png")))
        self.sidebar_button_3.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.histo_menu_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.histo_menu_frame.grid(row=6, column=0, padx=20, sticky="ew", pady=10)
        self.histo_menu_frame.grid_remove()

        self.histo_bt1 = ctk.CTkButton(self.histo_menu_frame, command=self.apply_histogram_stretching,
                                       text="Histogram Stretching", width=70, height=25,
                                       fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.histo_bt1.grid(row=0, column=0, sticky="ew", pady=2, padx=10)

        self.histo_bt2 = ctk.CTkButton(self.histo_menu_frame, command=self.apply_histogram_equalization,
                                       text="Histogram Equalization", width=70, height=25,
                                       fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.histo_bt2.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        # Neighborhood Processing Section
        self.neighbor_radio_var = ctk.StringVar(value="")  # Group neighbor radio buttons
        self.sidebar_button_4 = ctk.CTkButton(self.sidebar_frame, text="Neighborhood", font=("arial", 18),
                                              fg_color="#008585", width=200, height=40,
                                              command=self.toggle_neighbor_menu, hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\square.png")))
        self.sidebar_button_4.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

        self.neighbor_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.neighbor_frame.grid(row=8, column=0, padx=20, sticky="ew", pady=10)
        self.neighbor_frame.grid_remove()

        self.nei_la1 = ctk.CTkLabel(self.neighbor_frame, text="LINEAR", font=("arial", 18, "bold"))
        self.nei_la1.grid(row=0, column=0, sticky="w", pady=2, padx=10)

        self.nei_bt1 = ctk.CTkRadioButton(self.neighbor_frame, text="Average",
                                          command=lambda: [self.average_filter(),
                                                           self.update_restoration_radio_colors("Average")],
                                          variable=self.neighbor_radio_var, value="Average", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt1.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        self.nei_bt2 = ctk.CTkRadioButton(self.neighbor_frame, text="Laplacian",
                                          command=lambda: [self.laplacian_filter(),
                                                           self.update_restoration_radio_colors("Laplacian")],
                                          variable=self.neighbor_radio_var, value="Laplacian", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt2.grid(row=2, column=0, sticky="ew", pady=2, padx=10)

        self.nei_la2 = ctk.CTkLabel(self.neighbor_frame, text="NON LINEAR", font=("arial", 18, "bold"))
        self.nei_la2.grid(row=3, column=0, sticky="w", pady=2, padx=10)

        self.nei_bt3 = ctk.CTkRadioButton(self.neighbor_frame, text="Maximum",
                                          command=lambda: [self.maximum_filter(),
                                                           self.update_restoration_radio_colors("Maximum")],
                                          variable=self.neighbor_radio_var, value="Maximum", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt3.grid(row=4, column=0, sticky="ew", pady=2, padx=10)

        self.nei_bt4 = ctk.CTkRadioButton(self.neighbor_frame, text="Minimum",
                                          command=lambda: [self.minimum_filter(),
                                                           self.update_restoration_radio_colors("Minimum")],
                                          variable=self.neighbor_radio_var, value="Minimum", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt4.grid(row=5, column=0, sticky="ew", pady=2, padx=10)

        self.nei_bt5 = ctk.CTkRadioButton(self.neighbor_frame, text="Median",
                                          command=lambda: [self.median_filter(),
                                                           self.update_restoration_radio_colors("Median")],
                                          variable=self.neighbor_radio_var, value="Median", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt5.grid(row=6, column=0, sticky="ew", pady=2, padx=10)

        self.nei_bt6 = ctk.CTkRadioButton(self.neighbor_frame, text="Mode",
                                          command=lambda: [self.mode_filter(),
                                                           self.update_restoration_radio_colors("Mode")],
                                          variable=self.neighbor_radio_var, value="Mode", fg_color="#74a892",
                                          hover_color="#74a892", font=("arial", 14))
        self.nei_bt6.grid(row=7, column=0, sticky="ew", pady=2, padx=10)

        # Image Restoration Section
        self.restoration_radio_var = ctk.StringVar(value="")  # Group restoration radio buttons
        self.sidebar_button_5 = ctk.CTkButton(self.sidebar_frame, text="Image Restoration", font=("arial", 18),
                                              fg_color="#008585", width=200, height=40,
                                              command=self.toggle_restoration_menu, hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\repair.png")))
        self.sidebar_button_5.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        self.restoration_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.restoration_frame.grid(row=10, column=0, padx=20, sticky="ew", pady=10)
        self.restoration_frame.grid_remove()

        self.res_la1 = ctk.CTkLabel(self.restoration_frame, text="SALT & PEPPER NOISE", font=("arial", 18, "bold"))
        self.res_la1.grid(row=0, column=0, sticky="w", pady=2, padx=10)

        self.res_bt1 = ctk.CTkRadioButton(self.restoration_frame, text="Average",
                                          command=lambda: [self.apply_sp_average(),
                                                           self.update_restoration_radio_colors("Average")],
                                          variable=self.restoration_radio_var, value="Average", fg_color="#74a892",
                                          hover_color="#74a892",
                                          font=("arial", 14))
        self.res_bt1.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        self.res_bt2 = ctk.CTkRadioButton(self.restoration_frame, text="Median",
                                          command=lambda: [self.apply_sp_median(),
                                                           self.update_restoration_radio_colors("Median")],
                                          variable=self.restoration_radio_var, value="Median", fg_color="#74a892",
                                          hover_color="#74a892",
                                          font=("arial", 14))
        self.res_bt2.grid(row=3, column=0, sticky="ew", pady=2, padx=10)

        self.res_bt3 = ctk.CTkRadioButton(self.restoration_frame, text="Outlier Method",
                                          command=lambda: [self.apply_sp_outlier(),
                                                           self.update_restoration_radio_colors("Outlier Method")],
                                          variable=self.restoration_radio_var, value="Outlier Method",
                                          fg_color="#74a892", hover_color="#74a892",
                                          font=("arial", 14))
        self.res_bt3.grid(row=4, column=0, sticky="ew", pady=2, padx=10)

        self.res_la2 = ctk.CTkLabel(self.restoration_frame, text="GAUSSIAN NOISE", font=("arial", 18, "bold"))
        self.res_la2.grid(row=5, column=0, sticky="w", pady=2, padx=10)

        self.res_bt4 = ctk.CTkRadioButton(self.restoration_frame, text="Image Averaging",
                                          command=lambda: [self.apply_gaussian_image_averaging(),
                                                           self.update_restoration_radio_colors("Image Averaging")],
                                          variable=self.restoration_radio_var, value="Image Averaging",
                                          fg_color="#74a892", hover_color="#74a892",
                                          font=("arial", 14))
        self.res_bt4.grid(row=6, column=0, sticky="ew", pady=2, padx=10)

        self.res_bt5 = ctk.CTkRadioButton(self.restoration_frame, text="Average Filter",
                                          command=lambda: [self.apply_gaussian_average_filter(),
                                                           self.update_restoration_radio_colors("Average Filter")],
                                          variable=self.restoration_radio_var, value="Average Filter",
                                          fg_color="#74a892", hover_color="#74a892",
                                          font=("arial", 14))
        self.res_bt5.grid(row=7, column=0, sticky="ew", pady=2, padx=10)

        # Image Segmentation Section
        self.sidebar_button_6 = ctk.CTkButton(self.sidebar_frame, text="Image segmentation", font=("arial", 18),
                                              fg_color="#008585", width=200, height=40,
                                              command=self.toggle_segmentation_menu, hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\jigsaw.png")))
        self.sidebar_button_6.grid(row=11, column=0, padx=20, pady=10, sticky="ew")

        self.segmentation_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.segmentation_frame.grid(row=12, column=0, padx=20, sticky="ew", pady=10)
        self.segmentation_frame.grid_remove()

        self.seg_bt1 = ctk.CTkButton(self.segmentation_frame, text="Basic Global",
                                     command=self.basic_global_thresholding, width=70, height=25, fg_color="#74a892",
                                     corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.seg_bt1.grid(row=0, column=0, sticky="ew", pady=2, padx=10)

        self.seg_bt2 = ctk.CTkButton(self.segmentation_frame, text="Automatic", command=self.automatic_thresholding,
                                     width=70, height=25, fg_color="#74a892",
                                     corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.seg_bt2.grid(row=1, column=0, sticky="ew", pady=2, padx=10)

        self.seg_bt3 = ctk.CTkButton(self.segmentation_frame, command=self.adaptive_thresholding, text="Adaptive",
                                     width=70, height=25, fg_color="#74a892",
                                     corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.seg_bt3.grid(row=2, column=0, sticky="ew", pady=2, padx=10)
        self.seg_bt4 = ctk.CTkButton(self.segmentation_frame, command=self.enhance_colors, text="Enhance Colors",
                                     width=70, height=25,
                                     fg_color="#74a892", corner_radius=50, hover_color="#254e40", font=("arial", 14))
        self.seg_bt4.grid(row=3, column=0, sticky="ew", pady=2, padx=10)

        # Edge Detection Section
        self.sidebar_button_7 = ctk.CTkButton(self.sidebar_frame, text="Edge Detection", font=("arial", 18),
                                              fg_color="#008585", command=self.apply_sobel_edge_detection, width=200,
                                              height=40, hover_color="#004343",image=ctk.CTkImage(Image.open(r"C:\Users\ascom\Downloads\corner.png")))
        self.sidebar_button_7.grid(row=13, column=0, padx=20, pady=10, sticky="ew")

        self.sidebar_button_8=ctk.CTJ(self.sidebar_frame, text ="asmahan",font)

    def change_red_intensity(self, image, factor=1.5):
        modified = image.copy()
        modified[:, :, 2] = np.clip(modified[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return modified

    def swap_red_green(self, image):
        swapped = image.copy()
        swapped[:, :, [1, 2]] = swapped[:, :, [2, 1]]
        return swapped

    def eliminate_red(self, image):
        eliminated = image.copy()
        eliminated[:, :, 2] = 0
        return eliminated

    def to_rgb(self, image):
        rgb_img = np.zeros_like(image)
        rgb_img[:, :, 0] = image[:, :, 2]
        rgb_img[:, :, 1] = image[:, :, 1]
        rgb_img[:, :, 2] = image[:, :, 0]
        return rgb_img

    def update_restoration_radio_colors(self, selected_value):
        # Update colors for restoration radio buttons
        radio_buttons = {
            "Average": self.res_bt1,
            "Median": self.res_bt2,
            "Outlier Method": self.res_bt3,
            "Image Averaging": self.res_bt4,
            "Average Filter": self.res_bt5
        }
        
        for value, button in radio_buttons.items():
            if value == selected_value:
                button._fg_color = "#254e40"  # Selected color
            else:
                button._fg_color = "#74a892"  # Default color
            button._render()

    def apply_change_red(self):
        if self.image is not None:
            result = self.change_red_intensity(self.image)
            self.display_images(self.image, result, "Original", "change red intensity")

    def apply_swap_red_green(self):
        if self.image is not None:
            result = self.swap_red_green(self.image)
            self.display_images(self.image, result, "Original", "swap red and green")

    def apply_eliminate_red(self):
        if self.image is not None:
            result = self.eliminate_red(self.image)
            self.display_images(self.image, result, "Original", "eliminate red")

    def apply_to_rgb(self):
        if self.image is not None:
            result = self.to_rgb(self.image)
            self.display_images(self.image, result, "Original", "rgb")

    # Image Processing Methods
    def sobel_edge_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = np.uint8(sobel_combined)
        return sobel_combined

    def apply_sobel_edge_detection(self):
        if self.image is not None:
            result = self.sobel_edge_detection(self.image)
            self.display_images(self.image, result, "Original", "Sobel Edge")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = np.array(Image.open(file_path))
            self.display_images(self.image, self.image, "Original", "Original")

    def display_images(self, input_img, output_img, input_title, output_title):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.fig = plt.Figure(figsize=(8, 4))
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        if len(input_img.shape) == 3:
            ax1.imshow(input_img)
            ax2.imshow(output_img)
        else:
            ax1.imshow(input_img, cmap='gray')
            ax2.imshow(output_img, cmap='gray')

        ax1.set_title(input_title)
        ax2.set_title(output_title)
        ax1.axis('off')
        ax2.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # Menu Toggle Methods
    def toggle_point_menu(self):
        if self.point_menu_frame.winfo_ismapped():
            self.point_menu_frame.grid_remove()
        else:
            self.point_menu_frame.grid(row=2, column=0, padx=20, sticky="ew", pady=10)

    def toggle_color_menu(self):
        if self.color_menu_frame.winfo_ismapped():
            self.color_menu_frame.grid_remove()
        else:
            self.color_menu_frame.grid(row=4, column=0, padx=20, sticky="ew", pady=10)

    def toggle_histo_menu(self):
        if self.histo_menu_frame.winfo_ismapped():
            self.histo_menu_frame.grid_remove()
        else:
            self.histo_menu_frame.grid(row=6, column=0, padx=20, sticky="ew", pady=10)

    def toggle_neighbor_menu(self):
        if self.neighbor_frame.winfo_ismapped():
            self.neighbor_frame.grid_remove()
        else:
            self.neighbor_frame.grid(row=8, column=0, padx=20, sticky="ew", pady=10)

    def toggle_restoration_menu(self):
        if self.restoration_frame.winfo_ismapped():
            self.restoration_frame.grid_remove()
        else:
            self.restoration_frame.grid(row=10, column=0, padx=20, sticky="ew", pady=10)

    def toggle_segmentation_menu(self):
        if self.segmentation_frame.winfo_ismapped():
            self.segmentation_frame.grid_remove()
        else:
            self.segmentation_frame.grid(row=12, column=0, padx=20, sticky="ew", pady=10)

    def apply_add_brightness(self):
         if self.image is not None:
            brightness = int(self.slider.get() * 2.55)  # Convert slider value (0-100) to brightness (0-255)
            result = cv2.add(self.image, np.ones_like(self.image) * brightness)
            self.display_images(self.image, result, "Original", "Brightness Added")

    def apply_subtract_brightness(self):
        if self.image is not None:
            darkness = int(self.slider.get() * 2.55)
            result = cv2.subtract(self.image, np.ones_like(self.image) * darkness)
            self.display_images(self.image, result, "Original", "Brightness Subtracted")

    def apply_divide_image(self):
        if self.image is not None:
            factor = (self.slider.get() / 50.0) + 0.1  # Scale slider value to range 0.1-2.1
            result = (self.image / factor).astype(np.uint8)
            self.display_images(self.image, result, "Original", "Image Divided")

    def apply_complement_image(self):
        if self.image is not None:
            result = 255 - self.image
            self.display_images(self.image, result, "Original", "Image Complement")

    def apply_filter_rgb(self, img):
        b, g, r = cv2.split(img)
        filtered = cv2.merge([cv2.filter2D(ch, -1, self.kernel) for ch in (b, g, r)])
        self.display_images(img, filtered, "Original", "Filtered RGB")

    def outlier_method(self, img):
        cleaned = img.copy()
        h, w = img.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                neighborhood = img[i - 1:i + 2, j - 1:j + 2].flatten()
                med = np.median(neighborhood)
                if abs(int(img[i, j]) - med) > self.threshold:
                    cleaned[i, j] = med
        self.display_images(img, cleaned, "Original", "Outlier Removed")

    def apply_outlier_rgb(self, img):
        b, g, r = cv2.split(img)
        out_b = self._outlier_channel(b)
        out_g = self._outlier_channel(g)
        out_r = self._outlier_channel(r)
        result = cv2.merge([out_b, out_g, out_r])
        self.display_images(img, result, "Original", "Outlier RGB")

    def add_salt_pepper(self, prob=0.04, return_only=False):
        if self.image is None:
            print("No image loaded.")
            return None

        noisy = self.image.copy()
        h, w = self.image.shape[:2]
        salt_mask = np.random.rand(h, w) < prob / 2
        pepper_mask = np.random.rand(h, w) < prob / 2

        if len(self.image.shape) == 2:  # Grayscale
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
        else:  # Color (RGB, RGBA, etc.)
            channels = self.image.shape[2]
            noisy[salt_mask] = [255] * channels  # White
            noisy[pepper_mask] = [0] * channels  # Black

        if return_only:
            return noisy
        else:
            self.display_images(self.image, noisy, "Original", "Salt & Pepper Noise")
            return noisy

    def add_gaussian(self, mean=0, sigma=25, return_only=False):
        if self.image is None:
            print("No image loaded.")
            return None

        noise = np.random.normal(mean, sigma, self.image.shape).astype(np.float32)
        noisy = self.image.astype(np.float32) + noise
        result = np.clip(noisy, 0, 255).astype(np.uint8)

        if return_only:
            return result
        else:
            self.display_images(self.image, result, "Original", "Gaussian Noise")
            return result

    def my_cvtColor(self, code):
        if code == 'BGR2RGB':
            rgb_img = np.zeros_like(self.image)
            rgb_img[:, :, 0] = self.image[:, :, 2]
            rgb_img[:, :, 1] = self.image[:, :, 1]
            rgb_img[:, :, 2] = self.image[:, :, 0]
            return rgb_img
        elif code == 'BGR2GRAY':
            gray_img = np.dot(self.image[..., :3], [0.114, 0.587, 0.299])
            return gray_img.astype(np.uint8)
        else:
            raise ValueError("Unsupported conversion code")

    def convert_to_grayscale(self):
        return self.my_cvtColor('BGR2GRAY')

    def my_histogram(self):
        gray = self.convert_to_grayscale()
        hist = np.zeros(256, dtype=int)
        for value in gray.flatten():
            hist[value] += 1
        bins = np.arange(257)
        return hist, bins

    def apply_histogram_equalization(self):
        if self.image is not None:
            gray = self.convert_to_grayscale()
            hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            equalized_img = np.interp(gray.flatten(), bins[:-1], cdf_normalized).reshape(gray.shape).astype(np.uint8)
            self.display_images(gray, equalized_img, "Grayscale", "Equalized Histogram")

    def apply_histogram_stretching(self):
        if self.image is not None:
            gray = self.convert_to_grayscale()
            min_val = np.min(gray)
            max_val = np.max(gray)
            stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            self.display_images(gray, stretched, "Grayscale", "Stretched Histogram")

    def average_filter(self):
        if self.image is not None:
            avg_k = np.ones((3, 3), np.float32) / 9
            result = cv2.filter2D(self.image, -1, avg_k)
            self.display_images(self.image, result, "Original", "Average Filter")

    def maximum_filter(self):
        if self.image is not None:
            result = cv2.dilate(self.image, np.ones((3, 3), np.uint8))
            self.display_images(self.image, result, "Original", "Maximum Filter")

    def minimum_filter(self):
        if self.image is not None:
            result = cv2.erode(self.image, np.ones((3, 3), np.uint8))
            self.display_images(self.image, result, "Original", "Minimum Filter")

    def median_filter(self):
        if self.image is not None:
            result = cv2.medianBlur(self.image, 3)
            self.display_images(self.image, result, "Original", "Median Filter")

    def mode_filter(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            result = np.zeros_like(gray)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    neighborhood = gray[i - 1:i + 2, j - 1:j + 2].flatten()
                    values, counts = np.unique(neighborhood, return_counts=True)
                    result[i, j] = values[np.argmax(counts)]
            self.display_images(self.image, result, "Original", "Mode Filter")

    def update_radio_colors(self, selected_text):
        # Default color for unselected radio buttons
        default_color = "#FCFCFA"
        # Selected color
        selected_color = "#254e40"

        # Reset all radio buttons to default color
        for btn in [self.nei_bt1, self.nei_bt2, self.nei_bt3, self.nei_bt4, self.nei_bt5, self.nei_bt6]:
            btn.configure(fg_color=default_color, hover_color=default_color)

        # Set the selected radio button to the selected color
        if selected_text == "Average":
            self.nei_bt1.configure(fg_color=selected_color, hover_color=selected_color)
            self.average_filter()
        elif selected_text == "Laplacian":
            self.nei_bt2.configure(fg_color=selected_color, hover_color=selected_color)
            self.laplacian_filter()
        elif selected_text == "Maximum":
            self.nei_bt3.configure(fg_color=selected_color, hover_color=selected_color)
            self.maximum_filter()
        elif selected_text == "Minimum":
            self.nei_bt4.configure(fg_color=selected_color, hover_color=selected_color)
            self.minimum_filter()
        elif selected_text == "Median":
            self.nei_bt5.configure(fg_color=selected_color, hover_color=selected_color)
            self.median_filter()
        elif selected_text == "Mode":
            self.nei_bt6.configure(fg_color=selected_color, hover_color=selected_color)
            self.mode_filter()

    def laplacian_filter(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            result = cv2.Laplacian(gray, cv2.CV_64F)
            result = np.uint8(np.absolute(result))
            self.display_images(self.image, result, "Original", "Laplacian Filter")

    def image_averaging(self, num_images=5):
        averaged = np.zeros_like(self.image, dtype=np.float32)
        for _ in range(num_images):
            noisy = self.add_gaussian(self.image)
            averaged += noisy / num_images
        return averaged.astype(np.uint8)

    def basic_global_thresholding(self, thresh=127):
        """Basic global thresholding"""
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
            self.display_images(self.image, result, "Original", "Global Thresholding")

    def automatic_thresholding(self):
        """Automatic thresholding using Otsu's method"""
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
            self.display_images(self.image, result, "Original", "Otsu Thresholding")

    def adaptive_thresholding(self):
       if self.image is not None:
        # Convert to grayscale if image is color
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
            
        # Get image dimensions
        height = gray.shape[0]
        
        # Divide the image horizontally into two parts
        upper_half = gray[:height//2, :]
        lower_half = gray[height//2:, :]
        
        # Apply different thresholds to each half
        upper_threshold = cv2.adaptiveThreshold(upper_half, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        lower_threshold = cv2.adaptiveThreshold(lower_half, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 4)
        
        # Combine the two halves
        result = np.vstack((upper_threshold, lower_threshold))
        
        # Display the results
        self.display_images(self.image, result, "Original", "Adaptive Thresholding")

    def outlier_filter(self, img):
        if img is None:
            print("No image provided for outlier filter.")
            return None

        # Ensure img is single-channel (grayscale)
        if len(img.shape) != 2:
            print("Outlier filter expects a single-channel image.")
            return None

        cleaned = img.copy()
        h, w = img.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                neighborhood = img[i - 1:i + 2, j - 1:j + 2].flatten()
                med = np.median(neighborhood)
                if abs(int(img[i, j]) - med) > 50:
                    cleaned[i, j] = med
        return cleaned

    def apply_sp_average(self):
        if self.image is None:
            print("No image loaded.")
            return

        noisy = self.add_salt_pepper(prob=0.04, return_only=True)
        if noisy is None:
            return

        # Apply average filter
        avg_k = np.ones((3, 3), np.float32) / 9
        if len(noisy.shape) == 2:  # Grayscale
            result = cv2.filter2D(noisy, -1, avg_k)
        else:  # Color (RGB, RGBA, etc.)
            channels = cv2.split(noisy)
            result = cv2.merge([cv2.filter2D(ch, -1, avg_k) for ch in channels])

        self.display_images(self.image, result, "Original", "S&P + Average Filter")

    def apply_sp_median(self):
        if self.image is None:
            print("No image loaded.")
            return

        noisy = self.add_salt_pepper(prob=0.04, return_only=True)
        if noisy is None:
            return

        # Apply median filter
        if len(noisy.shape) == 2:  # Grayscale
            result = cv2.medianBlur(noisy, 3)
        else:  # Color (RGB, RGBA, etc.)
            channels = cv2.split(noisy)
            result = cv2.merge([cv2.medianBlur(ch, 3) for ch in channels])

        self.display_images(self.image, result, "Original", "S&P + Median Filter")

    def apply_sp_outlier(self):
        if self.image is None:
            print("No image loaded.")
            return

        noisy = self.add_salt_pepper(prob=0.04, return_only=True)
        if noisy is None:
            return

        # Apply outlier filter
        if len(noisy.shape) == 2:  # Grayscale
            result = self.outlier_filter(noisy)
        else:  # Color (RGB, RGBA, etc.)
            channels = cv2.split(noisy)
            processed_channels = [self.outlier_filter(ch) for ch in channels]
            # Check if any channel processing failed
            if any(ch is None for ch in processed_channels):
                print("Outlier filter failed for one or more channels.")
                return
            result = cv2.merge(processed_channels)

        if result is None:
            print("Outlier filter failed.")
            return

        self.display_images(self.image, result, "Original", "S&P + Outlier Method")

    def apply_gaussian_image_averaging(self):
        if self.image is None:
            print("No image loaded.")
            return

        # Generate multiple noisy images and average them
        num_images = 5
        averaged = np.zeros_like(self.image, dtype=np.float32)
        for _ in range(num_images):
            noisy = self.add_gaussian(mean=0, sigma=25, return_only=True)
            if noisy is None:
                return
            averaged += noisy.astype(np.float32) / num_images
        result = averaged.astype(np.uint8)

        self.display_images(self.image, result, "Original", "Gaussian + Image Averaging")

    def apply_gaussian_average_filter(self):
        if self.image is None:
            print("No image loaded.")
            return

        noisy = self.add_gaussian(mean=0, sigma=25, return_only=True)
        if noisy is None:
            return

        # Apply average filter
        avg_k = np.ones((3, 3), np.float32) / 9
        if len(noisy.shape) == 2:  # Grayscale
            result = cv2.filter2D(noisy, -1, avg_k)
        else:  # Color (RGB, RGBA, etc.)
            channels = cv2.split(noisy)
            result = cv2.merge([cv2.filter2D(ch, -1, avg_k) for ch in channels])

        self.display_images(self.image, result, "Original", "Gaussian + Average Filter")


if __name__ == "__main__":
    app = Web()
    app.mainloop()