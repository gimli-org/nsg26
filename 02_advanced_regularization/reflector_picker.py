import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def digitize_reflector(
    gpr_data,
    extent,
    output_file="reflector.txt",
    cmap="gray"
):
    """
    Open a Tkinter window to digitize a reflector on a GPR imshow.

    Parameters
    ----------
    gpr_data : 2D ndarray
        GPR image data
    extent : list or tuple
        [xmin, xmax, zmax, zmin] (important!)
    output_file : str
        Output text file path
    cmap : str
        Colormap for imshow
    """

    root = tk.Tk()
    root.title("GPR Reflector Digitizer")

    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.imshow(
        gpr_data,
        extent=extent,
        aspect="auto",
        cmap=cmap
    )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("m asl.")
    ax.set_title("Left click: add | Right click: undo | Enter: save")
    ax.set_aspect("equal")

    points = []

    line, = ax.plot([], [], "r-", lw=2)
    scatter = ax.scatter([], [], c="r", s=30)

    def update_plot():
        if points:
            x, z = zip(*points)
            line.set_data(x, z)
            scatter.set_offsets(np.column_stack([x, z]))
        else:
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
        canvas.draw()

    def onclick(event):
        if event.inaxes != ax:
            return

        # Left click
        if event.button == 1:
            points.append((event.xdata, event.ydata))
            update_plot()

        # Right click
        elif event.button == 3 and points:
            points.pop()
            update_plot()

    def onkey(event):
        if event.key == "enter":
            if not points:
                messagebox.showwarning("No data", "No points digitized.")
                return

            arr = np.array(points)
            np.savetxt(
                output_file,
                arr,
                fmt="%.6f",
                header="x z",
                comments=""
            )

            messagebox.showinfo(
                "Saved",
                f"{len(points)} points saved to:\n{output_file}"
            )

            root.destroy()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()