import tkinter as tk 
from tkinter import ttk
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     #writen with help from AI (14-04-26)
from datetime import datetime                                       #found with help from ai (16-04-26)
import time 

from LLM import call_llm as llm # this is just a test (keep as llm and switch out the rest)
    #Is the real LLM limited to only one sentance? maybe two for notifications? 

color_1 = "#6FBFA8" 
color_2 = "#8FE3C6"

FALLBACKS = {
    "a": "Your posture is being tracked — see the score above.",
    "b": "Your posture has shown variation over recent sessions.",
    "c": "Based on recent trends, continue wearing the vest to see improvement.",
}
NOTIF_FALLBACKS = {
    "a": "Reminder: check in on your posture — sit tall and roll your shoulders back.",
    "b": "Quick tip: stand up, stretch, and take a short walk to reset your posture.",
    "c": "Take care of your back today — limit time hunched over your phone or screen.",
}

def _summarize_values(value):
    """Turn a list of scores into a short English description."""
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    if not value:
        return "no data"
    avg     = sum(value) / len(value)
    recent  = value[-1]
    trend   = "improving" if value[-1] > value[0] else \
              "declining" if value[-1] < value[0] else "stable"
    return f"average {avg:.1f}, most recent {recent:.1f}, trend {trend}"

def stats_to_llm(inpu, value):
    if inpu == "a":
        input_LLM = "What is the current state of my posture?"
        llm_value = value[-1]            # current score, e.g. 4
    elif inpu == "b":
        input_LLM = "Make a summary of my postures current development"
        llm_value = value                # full list of recent scores
    elif inpu == "c":
        input_LLM = "Make a prediction of how my posture will develop in the future?"
        llm_value = value                # full list of recent scores

    out = llm(inpu=input_LLM, value=llm_value)
    if out.startswith("(No reply available"):
        return FALLBACKS[inpu]
    return out

def notification_from_llm(note_input, value):
    # Rotate a → b → c → a … robustly
    rotation = ["a", "b", "c"]
    if not note_input or note_input[-1] not in rotation:
        inpu = "a"
    else:
        inpu = rotation[(rotation.index(note_input[-1]) + 1) % 3]
    note_input.append(inpu)
    # Keep the list bounded so it doesn't grow forever
    if len(note_input) > 3:
        del note_input[:-3]

    if inpu == "a":
        input_LLM = "What is the current state of my posture?"
    elif inpu == "b":
        input_LLM = "Do you have a recomedation for something i can do right now to improve my posture?"
    elif inpu == "c":
        input_LLM = "How to watch out for my back?"

    out = llm(inpu=input_LLM, value=value[-1])
    if out.startswith("(No reply available"):
        out = NOTIF_FALLBACKS[inpu]

    return "\n" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + out + "\n"

class Statistics(ttk.Frame):
    def __init__(self, parent, data, data_today):
        super().__init__(parent, style="Custom.TFrame") 

        self.data_summary = data["Performance value"] #this is data over time
        self.data_today = data_today #current state 

        ttk.Label(self, text="Statistics", style="Custom.TLabel", font=("Arial", 17)).pack() 

        self.summary = tk.Text(self, height=15, width=80)
        self.summary.configure(bg=color_2) 
        self.summary.pack(pady=10)
        
        self.after_id = None
        self.update_summary()

        self.print_data(data)

    def update_summary(self):
        self.summary.delete("1.0", tk.END) #deleate already existing data/strings if there are

        #summary contain 3 main points 
        # 1. current state 
        self.summary.insert(tk.END, "Current state: \n  " + stats_to_llm(inpu = "a", value = self.data_today) + "\n")
        # 2. Development
        self.summary.insert(tk.END, "\nCurrent development: \n  " + stats_to_llm(inpu = "b", value = self.data_summary) + "\n") 
        # 3. Predict the future
        self.summary.insert(tk.END, "\nPrediction of the future: \n  " + stats_to_llm(inpu = "c", value = self.data_summary) + "\n")
        self.summary.insert(tk.END, "\n                                               Last updated:" + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        self.after_id = self.after(5000, self.update_summary) #after 5000 ms = 5sec, runs again after 5 sec 

    def print_data(self, data):
        data = data 

        date = data["Date"]
        performance = data["Performance value"]

        fig, ax = plt.subplots()
        ax.plot(date, performance, label="performance over time", color="green")
            
        for i, value in enumerate(performance):
            ax.text(date[i], value, f"{value:.1f}", ha="center", va="bottom")
            
        ax.legend()

        # the code below is wirtten with help from AI (14-04-26)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        plt.close(fig)

    def stop(self):
        if self.after_id:
            self.after_cancel(self.after_id)


class Notification(ttk.Frame):
    def __init__(self, parent, data_today): 
        super().__init__(parent, style="Custom.TFrame")

        self.data_today = data_today #current state 

        ttk.Label(self, text="Notification", style="Custom.TLabel", font=("Arial", 17)).pack() 

        self.notification = tk.Text(self, height=30, width=50)
        self.notification.configure(bg=color_2)
        self.notification.pack(pady=10)

        self.notification.insert(tk.END, "----- Note from LLM -----\n") 
        
        self.note_input = []
        self.after_id = None 
        self.new_notification()
    
    def new_notification(self):
        make_note = notification_from_llm(self.note_input, self.data_today)
        self.notification.insert(tk.END, make_note)
        self.after_id = self.after(5000, self.new_notification)

    def stop(self):
        if self.after_id:
            self.after_cancel(self.after_id)


def start_UI(data, data_today): 
    window = tk.Tk()
    window.title("HAPose")
    import os
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HApose.png")
    try:
        icon = tk.PhotoImage(file=icon_path)
        window.iconphoto(True, icon)
    except tk.TclError:
        pass  # icon file missing — not fatal
    window.geometry("1300x700")

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True) 

    style = ttk.Style() 
    style.theme_use("clam")
    style.configure("TNotebook.Tab", background=color_1)
    style.configure("Custom.TFrame", background=color_1)

    program_running = True

    stats_tab = Statistics(notebook, data, data_today)
    notif_tab = Notification(notebook, data_today)

    notebook.add(stats_tab, text="Statistics")
    notebook.add(notif_tab, text="Notification")

    def on_close():
        try:
            stats_tab.stop()
            notif_tab.stop()
        except:
            pass
        window.quit()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)

    exit_button = tk.Button(window, text = "Exit", command = on_close, font=("Times New Roman", 14), width = 5, height= 1)
    exit_button.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)

    window.mainloop()