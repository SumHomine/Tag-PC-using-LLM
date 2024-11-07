import os.path
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_shd_data(dir, cut_graph=True):
    data_shd = {
        "Dataset": ["Sprinkler", "Cancer", "Asia", "Sachs", "Child", "Insurance", "Alarm", "Barley", "Hepar2", "Win95PTS", "Andes"],
        "Standard PC": [2, 0, 3, 17, 12, 18, 4, 9, 9, 12, 10],
        "Typed PC (generic Prompt)": [2, 0, 3, 17, 8, 18, 9, 42, 7, 39, 97],
        "Tag PC - Tag Majority (generic Prompt)": [2, 0, 1, 17, 4, 0, 6, 9, 8, 7, 9],
        "Tag PC - Tag Weighted (generic Prompt)": [2, 0, 8, 17, 4, 4, 2, 9, 10, 13, 6],
        "Tag PC - Tag Majority (domain Prompt)": [None, None, 6, 7, None, 2, None, 12, None, 12, 9],
        "Tag PC - Tag Weighted (domain Prompt)": [None, None, 2, 13, None, 3, None, 8, None, 12, 9]
    }


    # Convert to a DataFrame
    df = pd.DataFrame(data_shd)

    # Set the dataset names as the index
    df.set_index("Dataset", inplace=True)


    custom_colors = [
        "brown",       # Standard PC
        "darkorange",    # Typed PC (LLM generated generic Prompt)
        "forestgreen",     # Tag PC - tag majority (LLM generated generic Prompt)
        "royalblue",   # Tag PC - tag weighted (LLM generated generic Prompt)
        "lightgreen",     # Tag PC - tag majority (LLM generated domain Prompt)
        "skyblue"    # Tag PC - tag weighted (LLM generated domain Prompt)
    ]

    # Plotting the data as a bar chart with custom colors
    fig, ax = plt.subplots(figsize=(16, 4))  # 5:1 aspect ratio
    df.plot(kind="bar", ax=ax, width=0.8, color=custom_colors)

    if cut_graph:
        # Cut chart of at 20
        ax.set_ylim(0, 20) #DELETE FOR FULL GRAPH

    # Set x-axis labels to be horizontal
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12) 

    ax.set_xlabel("", fontsize=14)  # Removes the "Dataset" label on the x-axis
    ax.set_ylabel("SHD", fontsize=14)     # Set y-axis label font size
    plt.title("SHD Comparison", fontsize=16)  # Set title font size

    # Display the legend outside the plot
    #plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left') #delete
    plt.legend(title="Algorithm", loc="upper left", fontsize=11, title_fontsize=14)

    # Save the plot as a PDF file
    plt.tight_layout()

    fname = "SHD_Bar_Chart" + ("" if cut_graph else "_uncircumcised") + ".pdf"
    plt.savefig(os.path.join(dir, fname), format="pdf")


def plot_sid_data(dir, cut_graph=True):
    data_sid = {
    "Dataset": ["Sprinkler", "Cancer", "Asia", "Sachs", "Child", "Insurance", "Alarm", "Barley"],
    "Standard PC": [9, 0, 18, 62, 228, 275, 46, 237],
    "Typed PC (generic Prompt)": [9, 0, 18, 62, 104, 233, 125, 972],
    "Tag PC - Tag Majority (generic Prompt)": [5, 0, 5, 62, 44, 0, 67, 237],
    "Tag PC - Tag Weighted (generic Prompt)": [5, 0, 28, 62, 24, 62, 27, 237],
    "Tag PC - Tag Majority (domain Prompt)": [None, None, 18, 24, None, 42, None, 205],
    "Tag PC - Tag Weighted (domain Prompt)": [None, None, 7, 39, None, 62, None, 89]
}


    # Convert to a DataFrame
    df = pd.DataFrame(data_sid)

    # Set the dataset names as the index
    df.set_index("Dataset", inplace=True)


    custom_colors = [
        "brown",       # Standard PC
        "darkorange",    # Typed PC (LLM generated generic Prompt)
        "forestgreen",     # Tag PC - tag majority (LLM generated generic Prompt)
        "royalblue",   # Tag PC - tag weighted (LLM generated generic Prompt)
        "lightgreen",     # Tag PC - tag majority (LLM generated domain Prompt)
        "skyblue"    # Tag PC - tag weighted (LLM generated domain Prompt)
    ]

    # Plotting the data as a bar chart with custom colors
    fig, ax = plt.subplots(figsize=(16, 4))  # 5:1 aspect ratio
    df.plot(kind="bar", ax=ax, width=0.8, color=custom_colors)

    if cut_graph:
        # Cut chart of at 20
        ax.set_ylim(0, 20) #DELETE FOR FULL GRAPH

    # Set x-axis labels to be horizontal
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12) 

    ax.set_xlabel("", fontsize=14)  # Removes the "Dataset" label on the x-axis
    ax.set_ylabel("SHD", fontsize=14)     # Set y-axis label font size
    plt.title("SHD Comparison", fontsize=16)  # Set title font size

    # Display the legend outside the plot
    #plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left') #delete
    plt.legend(title="Algorithm", loc="upper left", fontsize=11, title_fontsize=14)

    # Save the plot as a PDF file
    plt.tight_layout()

    fname = "SID_Bar_Chart" + ("" if cut_graph else "_uncircumcised") + ".pdf"
    plt.savefig(os.path.join(dir, fname), format="pdf")


dir = "Tag-PC-using-LLM/Experiment-Data/Experiment-Graphs-and-Tags"
plot_shd_data(dir=dir, cut_graph=True)